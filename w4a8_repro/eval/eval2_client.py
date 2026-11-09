#!/usr/bin/env python3
"""AIME2024/2025 + GPQA-diamond eval client - ais_bench protocol replica.

Prompts + golds are reused verbatim from the m3 (algo3_gptq_text) prediction
files, so prompt template / choice order are bit-identical to the original
AISBench runs. Sampling replicates the official ais_bench confs. Scoring:
AIME = last \\boxed{} int match; GPQA = last 'Answer: X' letter match.
--score-only validates the scorer against m3's saved predictions
(expect aime2024=90.0 aime2025=83.33 gpqa=79.8).
"""
import argparse
import asyncio
import json
import re
import time

REF = {
    "aime2024": "/home/w30022218/w4a8_tuning/aime/results/aime-m3-aisbench/outputs/default/20261029_222130/predictions/vllm-api-general-chat/aime2024.jsonl",
    "aime2025": "/home/w30022218/w4a8_tuning/aime/results/aime-m3-aisbench/outputs/default/20261030_010533/predictions/vllm-api-general-chat/aime2025.jsonl",
    "gpqa": "/home/w30022218/w4a8_tuning/gpqa/results/gpqa-m3-aisbench/outputs/default/20261030_044858/predictions/vllm-api-general-chat/GPQA_diamond.jsonl",
}
SAMPLING = {
    "aime": dict(temperature=1.0, top_p=0.95, top_k=20, seed=0, max_tokens=65536, batch=16),
    "gpqa": dict(temperature=0.6, top_p=0.95, seed=0, max_tokens=32768, batch=32),
}


def load_ref(bench):
    rows = []
    with open(REF[bench]) as f:
        for line in f:
            r = json.loads(line)
            prompt = "".join(m["prompt"] for m in r["origin_prompt"])
            rows.append({"id": r["id"], "prompt": prompt, "gold": r["gold"]})
    return rows


def strip_think(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def extract_boxed(text):
    idx = text.rfind("\\boxed")
    while idx != -1:
        i = text.find("{", idx)
        if i == -1:
            idx = text.rfind("\\boxed", 0, idx)
            continue
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1:j]
            j += 1
        idx = text.rfind("\\boxed", 0, idx)
    return None


def norm_int(s):
    if s is None:
        return None
    s = s.strip().strip("$").replace(",", "").replace(" ", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = re.sub(r"^\+", "", s)
    try:
        return int(s)
    except ValueError:
        m = re.search(r"-?\d+", s)
        return int(m.group()) if m else None


def extract_letter(text):
    ms = re.findall(r"Answer:\s*\**\s*([A-D])\b", text)
    return ms[-1] if ms else None


def score(bench, pred):
    text = strip_think(pred)
    if bench == "gpqa":
        return extract_letter(text)
    return norm_int(extract_boxed(text))


async def one(session, url, model, sp, row, sem, out, bench_ds):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": row["prompt"]}],
        "temperature": sp["temperature"],
        "top_p": sp["top_p"],
        "seed": sp.get("seed", 0),
        "max_tokens": sp["max_tokens"],
    }
    if "top_k" in sp:
        payload["top_k"] = sp["top_k"]
    async with sem:
        t0 = time.time()
        try:
            async with session.post(url + "/v1/chat/completions", json=payload) as resp:
                data = await resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            usage = data.get("usage", {})
            pred = score(bench_ds, content)
            goldv = row["gold"] if bench_ds == "gpqa" else norm_int(str(row["gold"]))
            rec = dict(row)
            rec.update(prediction=content, extracted=str(pred),
                       correct=(pred == goldv),
                       out_tokens=usage.get("completion_tokens"),
                       seconds=round(time.time() - t0, 1))
            out.append(rec)
            print("[%s] id=%s gold=%s pred=%s %s (%s tok, %ss)" % (
                bench_ds, row["id"], row["gold"], pred,
                "OK" if rec["correct"] else "MISS",
                rec["out_tokens"], rec["seconds"]), flush=True)
        except Exception as e:  # noqa: BLE001
            out.append(dict(row, prediction="", extracted=None, correct=False,
                            error=repr(e)[:150], seconds=round(time.time() - t0, 1)))
            print("[%s] id=%s ERROR %r" % (bench_ds, row["id"], e), flush=True)


async def run_bench(session, url, model, bench_ds, sp, tag, out_dir):
    sem = asyncio.Semaphore(sp["batch"])
    out = []
    rows = load_ref(bench_ds)
    await asyncio.gather(*[
        one(session, url, model, sp, r, sem, out, bench_ds) for r in rows])
    ok = sum(1 for r in out if r["correct"])
    acc = 100.0 * ok / max(1, len(out))
    with open("%s/eval2-%s-%s-predictions.jsonl" % (out_dir, tag, bench_ds), "w") as f:
        for r in sorted(out, key=lambda x: x["id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return acc, ok, len(out)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["aime", "gpqa"], required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="qwen36-eval2")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="/home/w30022218/w4a8_tuning/ceval/results")
    ap.add_argument("--score-only", help="validate scorer on a saved m3 predictions jsonl")
    ap.add_argument("--only", help="run only this dataset (e.g., aime2025)")
    args = ap.parse_args()

    if args.score_only:
        bench = "gpqa" if "gpqa" in args.score_only.lower() else "aime"
        rows = [json.loads(l) for l in open(args.score_only)]
        ok = 0
        for r in rows:
            pred = score(bench, r.get("prediction", ""))
            goldv = r["gold"] if bench == "gpqa" else norm_int(str(r["gold"]))
            if pred == goldv:
                ok += 1
            else:
                print("MISS id=%s gold=%s extracted=%s" % (r["id"], r["gold"], pred))
        print("SCORE-ONLY %s: %d/%d = %.2f%%" % (
            args.score_only.split("/")[-1], ok, len(rows), 100.0 * ok / len(rows)))
        return

    import aiohttp
    sp = SAMPLING[args.bench]
    datasets = ["aime2024", "aime2025"] if args.bench == "aime" else ["gpqa"]
    if args.only:
        datasets = [d for d in datasets if d == args.only]
    url = "http://127.0.0.1:%d" % args.port
    timeout = aiohttp.ClientTimeout(total=None)
    t0 = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = {}
        for ds in datasets:
            acc, ok, n = await run_bench(session, url, args.model, ds, sp, args.tag, args.out_dir)
            results[ds] = acc
            print("== %s: %d/%d = %.2f%%" % (ds, ok, n, acc), flush=True)
    summary = {"tag": args.tag, "model": args.model, "port": args.port,
               "bench": args.bench, "sampling": dict(sp),
               "seconds": round(time.time() - t0, 1)}
    summary.update(results)
    if args.bench == "aime" and "aime2024" in results and "aime2025" in results:
        summary["mean"] = (results["aime2024"] + results["aime2025"]) / 2
    with open("%s/eval2-%s-summary.json" % (args.out_dir, args.tag), "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY " + json.dumps(summary))


asyncio.run(main())
