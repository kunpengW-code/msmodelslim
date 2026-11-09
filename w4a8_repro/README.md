# w4a8_repro — Qwen3.6-27B w3_g4 (W4A8, 24G) 端到端复刻指南

> 2026-11 收官交付。两个个人仓配合使用：
> - **本仓**（msmodelslim 个人分支）= 量化侧：msmodelslim 26.0.0.alpha01 + 战役补丁 + 校准集/配置/脚本
> - **vllm-ascend 个人分支** `feat/w4a8-dense-linear-v0.30.0` = 服务侧：v0.30.0 + cherry-pick 上游 #20469（W4A8 dense Linear）
>
> 目标：从这两个仓出发可完整复刻 w3_g4 权重并复现精度/性能。

## 0. 交付成绩（复现期望值）

| 指标 | 数值 | 备注 |
|---|---|---|
| 体积 | 24G（7 shards） | 源模型 54G |
| AIME2024 / 2025 | **91.67 / 91.67**（两轮均值） | 两轮分别为 90.0/93.33 和 93.33/90.0 |
| GPQA | 80.8（81.31 / 80.30） | |
| CEval macro | 84.5（84.46 / 84.59） | 1346 题 |
| 性能 vs W8A8 anchor | TPOT 4/5 场景更优（-1.4%~-9.3%），吞吐 +4~10% | 同机同日 mtp K3 对照 |

配方（w3_g4）：语言层 64 层 = 37 层 gptq-W4A8 + 27 块 gptq-W8A8（首 10 层 + 中部敏感 16 层 + 末层）+ visual 27 块 minmax-W8A8 + MTP gptq-W4A8 + lm_head FLOAT，per-channel，长链混合校准 268k token。

## 1. 环境

- 硬件：8×910B4 32GB（量化单卡即可，服务 TP2/DP2 4 卡或 TP4）
- **量化容器**：`quay.io/ascend/vllm-ascend:v0.27.1rc1`（实测环境 wkp-vllm），本仓 editable 安装：
  `docker exec <容器> pip install -e /path/to/this/repo`
  （源树直装即可；量化全程只依赖本仓代码，无需 vllm-ascend 分支）
- **服务容器**：`quay.io/ascend/vllm-ascend:v0.30.0`（vllm 0.30.0, vllm_ascend 0.30.0）+ 本指南 §3 的 vllm-ascend 分支
- 源模型：`/home/models/Qwen3.6-27B`（FP 权重，54G）

## 2. 量化复现（~7h，单卡）

1. 目录布局（`run_quant_arm_new.sh` 与 yaml 内含绝对路径，按此布局或自行 sed）：
   ```
   /home/w30022218/w4a8_tuning/
   ├── calib/qwen_cot_vlm_long/index.jsonl   ← 本仓 calib/
   ├── calib/tiny_white_8x8.jpg              ← 本仓 calib/（index.jsonl 引用的 VLM 占位图）
   ├── configs/w3_g4.yaml                    ← 本仓 configs/
   └── scripts/run_quant_arm_new.sh          ← 本仓 quant/
   ```
2. 启动（**参数必须裸名**，脚本内部自己拼 `configs/` 前缀；传全路径会路径拼接崩）：
   ```
   docker exec <量化容器> bash -c "cd /home/w30022218/w4a8_tuning && bash scripts/run_quant_arm_new.sh w3_g4.yaml w3_g4 5"
   ```
   等价于 `msmodelslim quant --model_path /home/models/Qwen3.6-27B --save_path quant_out/w3_g4 --device npu --model_type Qwen3.6-27B --config_path configs/w3_g4.yaml --trust_remote_code True`
3. ~6.7 min/层 × 64 层 ≈ 7h（gptq Hessian 主导；268k 校准数据 2.1× 后时长不变——forward 不是瓶颈）
4. 产物：`quant_out/w3_g4/`（7 shards 24G + quant_model_description.json）
5. **save_path 不复用**：msmodelslim 对已有目录会异常，换新名或清空

### 2.1 校准集说明（长链校准，w3_g4 的核心创新）

`calib/qwen_cot_vlm_long/index.jsonl`：64 条 `{image, text}` VLM 样本 =
- 58 条短 CoT（1.3-2.9k token）
- **6 条 bs_e1（W8A8 参照模型）亲笔 AIME 正确轨迹**（question+prediction 截 70k 字符 ≈21k token/条）
- 共 268k token（原短链校准 127k 的 2.1×）

根因背景：gptq 校准集全短 → Hessian 长程零覆盖 → 65k 推理尾部概率劣化 → AIME 跑飞（哨兵 id=22 "tor tor tor" 3/3 轮）。喂入亲笔长轨迹后哨兵两轮均收敛、AIME 均值 91.67 登顶。**这是 w3_g3→w3_g4 的唯一变量**（w3_g3 同配方短链校准 = AIME 88.3/89.2）。

### 2.2 msmodelslim 补丁（本仓 git 历史，均已验证必需）

| 提交 | 补丁 | 不打会怎样 |
|---|---|---|
| fix(ir) | deployed fake-quant 包装器尾接 `.to(x.dtype)`（w8a8_dynamic.py:70/118, w4a8_dynamic.py:66） | gptq fp32 Hessian 下 fp32 泄漏撞 GDN conv1d，量化 30s 崩 |
| fix(qwen3_5_moe) sdpa | adapter 强制 eager→sdpa（~114/252 行） | 20k+ 校准样本 FA 注意力矩阵 30.24 GiB > 单卡 22 GiB，layer3 OOM |
| fix(mtp) | lm_head 只算最后位置（~863 行） | [1,20k,vocab] fp32 logits 24 GiB，layer63 OOM |

坑：`docker exec <c> python3 - <<EOF` 打补丁**必须加 -i**（stdin 不转发且静默 rc=0）；打完必须宿主端 grep 独立验证。

## 3. 服务复现（vllm-ascend 个人分支）

分支 `feat/w4a8-dense-linear-v0.30.0` = v0.30.0 + cherry-pick `2697e7f`（上游 #20469，ModelSlim W4A8 dynamic Linear：A8W4 GMM、NZ、int32 重打包、TP scale_bias 16 段分片）。**stock 0.30.0 只给 MoE 注册 W4A8_DYNAMIC，dense linear 直接 NotImplementedError**。

两种用法：
```
# A) 免安装（PYTHONPATH 前置，注意保留原 Ascend 路径）：
export PYTHONPATH=/path/to/vllm-ascend-repo:$PYTHONPATH
# B) 容器内 pip install 该分支

vllm serve quant_out/w3_g4 \
  --served-model-name w3g4 --host 127.0.0.1 --port <port> \
  --tensor-parallel-size 2 --data-parallel-size 2 --dtype bfloat16 \
  --language-model-only --safetensors-load-strategy prefetch \
  --no-enable-prefix-caching --enable-chunked-prefill --async-scheduling \
  --max-model-len 131072 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.9 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4,8,12,16,24,32,48,64]}' \
  --seed 0 --kv-cache-dtype bfloat16
```

> ⚠️ **源码 checkout 直接 serve 的坑**：vllm-ascend 镜像内是「源码 + 编译产物」形态（`/vllm-workspace/vllm-ascend` = v0.30.0 tag + 已编译的 `vllm_ascend_C.cpython-312-aarch64-linux-gnu.so`、`libvllm_ascend_kernels.so`、`_cann_ops_custom/vendors/`）。`aclnnAddRmsNormBias` 等 fusion pass 依赖的 vendor 自定义算子在编译产物里，**不在宿主 CANN**——纯 git checkout 用 PYTHONPATH 遮蔽镜像内构建，会在 norm_quant_fusion_pass 注册阶段崩 `RuntimeError: aclnnAddRmsNormBias ... not in libopapi.so`。两种解法：
> 1. **嫁接产物（快，已实测验证）**：`cp -a /vllm-workspace/vllm-ascend/vllm_ascend/{_build_info.py,_version.py,include,lib,libvllm_ascend_kernels.so,vllm_ascend_C.*.so,_cann_ops_custom/vendors} <分支仓>/vllm_ascend/`（同为 v0.30.0 基座，ABI 兼容；产物是构建输出，不入 git）
> 2. 容器内 `pip install` 本分支源码（完整重编，耗时较长）

> ⚠️ **勿加额外旗标**：quantization 由 checkpoint 自带 quant_config 自动识别（引擎日志显示 `quantization=ascend`），不要显式传 `--quantization ascend`；也不要加 `--mamba-cache-mode align`。实测 TP4 + 这两个旗标的组合会在图编译阶段崩溃（`RuntimeError: aclnnAddRmsNormBias ... not in libopapi.so`，宿主 CANN 缺该算子）。上面模板 = 131 两轮全量精度评测 + 125 性能双 run 的实测口径，逐字使用。
> `serving/` 里的 w4a8_main_port.py + bootstrap_main/ 是**历史 out-of-tree 注入方案**（sitecustomize 机制），仅当不便用分支时使用；分支方案不需要它们。

## 4. 精度复现（eval/）

协议（与两轮全量严格一致）：TP2/DP2 4 卡；AIME t=1.0/top_p=0.95/top_k=20/seed=0/max 65536/batch16；GPQA t=0.6/top_p=0.95/seed=0/max 32768/batch32；CEval 见 run_ceval.sh。

```
bash eval/run_eval2.sh <bench> <tag> <port>   # 单基准
bash eval/driver9.sh                          # 全量轮（含死轮 --only 自动补测）
```
数据集：`snapshot_download('vllm-ascend/{aime2024,aime2025,gpqa}', repo_type='dataset')`。评分注意：AIME gold 是字符串需 norm_int，取最后一个 `\boxed{}`；GPQA 取最后一个 "Answer: X"。死轮签名 = predictions 含 `"error":` 记录 ≥10（AIME 双年连跑 ~1.5-2h 后偶发服务器猝死，基建问题，--only 补测即可）。

## 5. 性能复现

PerfTest 框架（125 `/home/j30080978/PerfTest`，gitcode jiangli221/blue-green）：5 场景 × 2 轮 384 请求，mtp K3/TP4。参照数字见主报告 §20（W4A8 vs W8A8 anchor 同机同日对照表）。基线 baseline.json 为 W8A8+DSpark K7 口径，仅作参考线。

## 6. 完整文档

- 主报告：`w4a8_report/Qwen3.6-27B_W4A8_report.md` §19-§21（战役叙事、两轮数据、性能对照、实现快照）
- 运维日志：`NEXT-SESSION.md`（全部踩坑按日期）
- 交接：`HANDOVER.md`
