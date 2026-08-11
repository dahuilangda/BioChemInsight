# MolNexTR MoE 训练

BioChemInsight 的 MolNexTR 混合专家（MoE）训练流程，用于识别完整分子、Markush 结构和带附着点的片段。

## 部署

`constants.py` 的 `MOLNEXTR_MOE_CONFIG_PATH` 指向当前 MoE 配置。留空则回退到冻结的基础模型（无 MoE）。

```python
MOLNEXTR_MOE_CONFIG_PATH = 'experiments/moe/molnextr_v5_native_full_valence_v3/moe_config.json'
```

## 数据流水线

```bash
bash training/molnextr_markush/scripts/run_moe_production.sh [stage]
```

| 阶段 | 说明 |
|------|------|
| `generate` | 生成 ordinary/markush/fragment 分片 |
| `qc` | 源数据校验与覆盖率检查 |
| `build-data` | 合并分片为训练 dataframe |
| `train` | MoE 训练（支持 DDP） |
| `eval` | held-out + 真实 MarkushGrapher + 真实难例评估 |

数据源：

```
training/molnextr_markush/data/generated/pose_factory/molnextr_moe_production_v1/
training/molnextr_markush/data/generated/real_markushgrapher_ocsr_v2/{train,eval}/data.parquet
experiments/moe/molnextr_moe_production_v1_train_df.parquet
```

## 训练

### 生产入口

```bash
# QC
bash training/molnextr_markush/scripts/run_moe_production.sh --stage qc

# 训练（2 卡）
bash training/molnextr_markush/scripts/run_moe_production.sh --stage train --reuse-df-cache 0 --ddp-gpus 2

# 训练（单卡）
bash training/molnextr_markush/scripts/run_moe_production.sh --stage train --reuse-df-cache 0 --ddp-gpus 1
```

重新生成数据后必须加 `--reuse-df-cache 0`。

### 默认超参数

```
epochs=12  batch-size=8  grad-accum=2
lr=5e-5  encoder-lr=1e-5  encoder-finetune-stages=1
router-lr=1e-4  attachment-set-lr=2e-4
routing-strategy=soft_mixture
expert-kind=full_mixture  full-mixture-sidecar-mode=per_sidecar
attachment-set-decode-mode=direct_sidecar
attachment-set-feature-mode=multiscale_pointer_heatmap
real-original-weight=12.0  sampling-focus-fraction=0.10
```

### 从 checkpoint 续训

```bash
python -m torch.distributed.run --nproc_per_node=2 \
  training/molnextr_markush/tools/train_moe.py \
  --epochs 12 --batch-size 8 --grad-accum 2 \
  --lr 2e-5 --encoder-lr 2e-6 --encoder-finetune-stages 2 \
  --resume-encoder <model_dir>/moe_encoder.pth \
  --resume-expert1 <model_dir>/moe_expert1.pth \
  --resume-expert2 <model_dir>/moe_expert2.pth \
  --resume-router <model_dir>/moe_router.pt \
  --edge-valence-loss-weight 0.5 \
  ...其他参数...
```

`--edge-valence-loss-weight`：可微的 expected-valence 惩罚，教 decoder 不画超价图。0 关闭，推荐 0.5。

## 评估

```bash
bash training/molnextr_markush/scripts/run_moe_production.sh --stage eval
```

| 脚本 | 评估集 |
|------|--------|
| `eval_moe.py` | held-out complete/markush/fragment |
| `eval_real_markushgrapher.py` | 1195 行真实专利图像 |
| `eval_real_wavy_hard_eval.py` | 32 行手工标注难例 |

仅跑 held-out：

```bash
bash training/molnextr_markush/scripts/run_moe_production.sh \
  --stage eval --run-real-markush-eval 0 --run-real-hard-eval 0 --run-real-task-eval 0
```

### 安全测试

```bash
python tests/test_moe_training.py
python tests/test_moe_byte_identical.py
python tests/test_moe_production_contracts.py
python tests/test_moe_attachment_set.py
```

## 架构

### MoE 结构

- **Expert 0**（complete）：冻结的原始 MolNexTR decoder，普通分子识别 byte-identical 保留。
- **Expert 1**（markush）：可训练 sidecar decoder，直接输出 Markush 图（含 dummy 原子和边）。
- **Expert 2**（fragment）：可训练 sidecar decoder，识别带附着点的片段。
- **Router**：attention-pool 路由器。推理时优先使用显式结构类型路由。

### Attachment-set Head

DETR 风格 query head，预测附着点位置、数量、键合状态和键类型。使用 Hungarian 匹配（完整集）或 positive-unlabeled 匹配（部分观测集）。辅助诊断任务，不修改图。

### 化学层修复

`utils/MolNexTR/chemical.py` 中图→SMILES 转换的后处理：

1. **超价修复**（`repair_hypervalent_edges`）：从超价原子上删优先级最低的键（dummy > 低键级），直到图通过 sanitize。对合法图是 no-op。
2. **孤立 dummy 重连**（`re_bond_orphan_dummies`）：修复产生的孤立 dummy 按坐标距离重连到有剩余价的原子上。

## 目录结构

```
training/molnextr_markush/
├── scripts/run_moe_production.sh   生产编排
├── src/                            核心数据库
│   ├── moe_dataset.py              dataframe 构建
│   ├── moe_sources.py              源数据发现
│   ├── pose_factory.py             数据生成
│   └── sampling.py                 采样策略
├── tools/                          训练 + 数据工具
│   ├── train_moe.py                MoE 训练器
│   ├── check_*.py                  预检 / 契约校验
│   ├── build_*.py                  数据 / 分片构建
│   └── audit_*.py                  QA / 审计
└── synth/                          合成数据渲染

evaluation/
├── eval_moe.py
├── eval_real_markushgrapher.py
└── eval_real_wavy_hard_eval.py
```

## 不变量

- 完整分子路径（Expert 0）byte-identical 保留。
- Sidecar 解码使用硬全图拥有权：Markush/fragment 专家直接输出完整图。
- Fragment 目标必须恰好包含一个键合的终端 dummy 原子。
- 真实专利训练行必须有验证的页面图像 pose。
