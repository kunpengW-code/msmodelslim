#!/usr/bin/env python3
"""C-Eval validation client — ported byte-faithfully from y00884608's
evaluate_ceval.py (ceval-w8a8-w4a8c8-attribution-20261020, 2026-10-20).

Protocol is IDENTICAL to the original so results are directly comparable with
the historical numbers (bf16 85.14% / w8a8 85.51% / w4a8 81.20% / ...):
  - prompt: 0-shot answer-only Chinese instruction + question + A/B/C/D + 答案：
  - request: temperature=0, max_tokens=32, chat_template_kwargs{enable_thinking:False}
  - concurrency: 8 (asyncio semaphore)
  - scoring: first standalone A/B/C/D letter; no match => invalid => wrong
  - summary JSON schema: same keys as the original

Only differences from the original (protocol-neutral):
  - CLI parameters instead of hardcoded paths (url/model/tag/data/out)
  - bounded transport retry (3 attempts) on connection errors / timeouts / 5xx
    so a transient network blip cannot kill a multi-hour run; the request body
    is unchanged
  - --limit N for smoke runs (first N questions only)
  - probability probes are best-effort: per-probe errors are recorded in the
    output instead of aborting the run

Usage:
  python3 ceval_client.py --tag c3_w4a8_mixed \
      [--url http://127.0.0.1:18089] [--model qwen36-ceval] \
      [--data data/validation.jsonl] [--out results] [--limit N] [--no-probes]
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import aiohttp

PROMPT_HEADER = '请回答下面的单项选择题。只输出正确选项的字母（A、B、C 或 D），不要解释。\n\n'

# 8 fixed probability-probe texts, identical to the original client.
PROBE_TEXTS = [
    '人工智能模型的性能取决于计算、存储和通信。量化可以减少权重和缓存占用，但需要验证精度。',
    'The capital of France is Paris. A computer executes instructions and stores data in memory.',
    '已知三角形的三条边长为3、4、5，则它是一个直角三角形，面积等于6。',
    'def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n',
    '在保持温度不变的条件下，理想气体的压强和体积成反比。',
    '北京、上海、广州和深圳都是中国的重要城市，各有不同的产业和文化特点。',
    'A fair coin has two possible outcomes. The probability of heads is one half.',
    '机器学习中，训练集用于优化参数，验证集用于选择配置，测试集用于独立评估。',
]

RETRY_ATTEMPTS = 3


async def post_json(session, url, body):
    """POST JSON with bounded transport retry; identical request body."""
    last_error = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            async with session.post(url, json=body) as response:
                payload = await response.json()
                if response.status == 200:
                    return payload
                last_error = RuntimeError(f'HTTP {response.status}: {payload}')
                if response.status < 500:
                    raise last_error  # client error, retrying will not help
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            last_error = error
        await asyncio.sleep(2 * (attempt + 1))
    raise last_error


async def evaluate(url, model, tag, data_path, out_dir, limit, probes_enabled):
    samples = [json.loads(line) for line in
               Path(data_path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if limit:
        samples = samples[:limit]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(8)
    completed = []
    started = time.time()
    with (out / f'ceval-{tag}-responses.jsonl').open('w', encoding='utf-8') as log:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
            async def one(sample):
                async with semaphore:
                    prompt = (PROMPT_HEADER + sample['question'] + '\n'
                              + '\n'.join(f'{c}. {sample[c]}' for c in 'ABCD')
                              + '\n答案：')
                    body = dict(model=model,
                                messages=[dict(role='user', content=prompt)],
                                temperature=0, max_tokens=32,
                                chat_template_kwargs=dict(enable_thinking=False))
                    payload = await post_json(session, url + '/v1/chat/completions', body)
                    content = payload['choices'][0]['message']['content'] or ''
                    matched = re.search(r'(?:^|[^A-Za-z])([ABCD])(?:[^A-Za-z]|$)', content)
                    prediction = matched.group(1) if matched else None
                    result = dict(sample_id=sample.get('sample_id', sample.get('id')),
                                  subject=sample['subject'], answer=sample['answer'],
                                  prediction=prediction, correct=prediction == sample['answer'],
                                  content=content, usage=payload.get('usage'),
                                  finish_reason=payload['choices'][0]['finish_reason'])
                    completed.append(result)
                    log.write(json.dumps(result, ensure_ascii=False) + '\n')
                    log.flush()
                    if len(completed) % 100 == 0:
                        print(tag, len(completed), flush=True)
            await asyncio.gather(*(one(s) for s in samples))

    subjects = {}
    for s in completed:
        group = subjects.setdefault(s['subject'], dict(total=0, correct=0))
        group['total'] += 1
        group['correct'] += s['correct']
    for s in subjects.values():
        s['accuracy'] = s['correct'] / s['total']
    result = dict(tag=tag, total=len(completed),
                  correct=sum(s['correct'] for s in completed),
                  invalid=sum(s['prediction'] is None for s in completed),
                  subjects=subjects,
                  protocol=dict(shots=0, temperature=0, max_tokens=32, enable_thinking=False,
                                scoring='first standalone A/B/C/D; invalid is wrong',
                                concurrency=8),
                  seconds=time.time() - started)
    result['accuracy'] = result['correct'] / result['total']
    result['macro_accuracy'] = sum(s['accuracy'] for s in subjects.values()) / len(subjects)

    if probes_enabled:
        probes = []
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
                for text in PROBE_TEXTS:
                    try:
                        payload = await post_json(session, url + '/v1/completions',
                                                  dict(model=model, prompt=text, temperature=0,
                                                       max_tokens=1, echo=True, logprobs=1))
                        probes.append(dict(prompt=text, response=payload))
                    except Exception as error:  # best-effort: record, never abort
                        probes.append(dict(prompt=text, error=repr(error)))
        except Exception as error:
            probes.append(dict(error=repr(error)))
        (out / f'probability-{tag}.json').write_text(
            json.dumps(probes, ensure_ascii=False, indent=2), encoding='utf-8')

    (out / f'ceval-{tag}-summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('tag', 'total', 'correct', 'invalid', 'accuracy', 'macro_accuracy', 'seconds')},
                     ensure_ascii=False))
    return result


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--url', default='http://127.0.0.1:18089')
    ap.add_argument('--model', default='qwen36-ceval')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--data', default=str(here / 'data' / 'validation.jsonl'))
    ap.add_argument('--out', default=str(here / 'results'))
    ap.add_argument('--limit', type=int, default=0, help='first N questions only (smoke)')
    ap.add_argument('--no-probes', action='store_true', help='skip probability probes')
    args = ap.parse_args()
    asyncio.run(evaluate(args.url, args.model, args.tag, args.data, args.out,
                         args.limit, not args.no_probes))


if __name__ == '__main__':
    main()
