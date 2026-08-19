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
# 冒烟测试（快，无需 GPU）
python -m training.molnextr_markush.cli smoke

# 完整契约测试
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
├── cli.py                          统一 CLI 入口
├── scripts/
│   ├── run_moe_production.sh       生产编排（generate/qc/build/train/eval）
│   ├── run_bond_finetune.sh        键级微调
│   └── download_data.py            数据集下载
├── src/                            核心库
│   ├── base_trainer.py             BaseTrainer + BaseCocoMaskRCNNTrainer（共享训练基类）
│   ├── rdkit_utils.py              RDKit 工具（懒加载）
│   ├── moe_dataset.py              dataframe 构建
│   ├── moe_sources.py              源数据发现
│   ├── pose_factory.py             pose-factory 校验契约
│   ├── sampling.py                 课程采样策略
│   ├── markush_layout_labels.py    Markush 布局标注
│   ├── labels.py / curriculum.py / schema.py / config.py
│   ├── molnextr_dataset_toolkit/   数据生成 pipeline（pipeline/config/contracts/retry）
│   └── moe_trainer/                MoE 训练器（从 train_moe.py 拆分）
│       ├── __init__.py             re-export 公开 API
│       ├── _common.py              共享 import + 路径设置
│       ├── args.py                 参数解析 + 契约校验
│       ├── model.py                EncoderMoETrainingModel + 微调配置
│       ├── losses.py               RLOO / DPO / terminal-action 损失
│       ├── data.py                 数据契约 + 分区加载
│       ├── calibration.py          sidecar/router 阈值校准
│       ├── runtime.py              LabelBalancedSampler / checkpoint / scheduler
│       └── entry.py                main() 训练循环
├── tools/                          可执行训练 + 数据工具
│   ├── train_moe.py                ← Shim → src/moe_trainer/
│   ├── train_wavy_maskrcnn.py      WavyMaskRCNNTrainer（class-based）
│   ├── train_attachment_maskrcnn.py AttachmentMaskRCNNTrainer（class-based）
│   ├── train_wavy_unet.py          WavyUNetTrainer（class-based）
│   ├── train_markush_layout_expert.py
│   ├── train_fragment_attachment_expert.py
│   ├── train_confidence_head.py
│   ├── calibrate_fragment_attachment_expert.py
│   ├── calibrate_markush_layout_expert.py
│   ├── build_pose_factory_*.py     pose-factory 分片构建
│   ├── build_*_splits.py           训练/验证拆分
│   ├── gen_*_seg_data.py           分割数据生成
│   └── ...                         用 `cli list` 查看全部
└── tests/
    └── smoke/                      冒烟测试（<10s，无需 GPU）
        ├── test_imports.py
        ├── test_train_moe_split.py
        ├── test_cli.py
        ├── test_data_contract.py
        └── test_base_trainer.py

evaluation/
├── eval_moe.py
├── eval_real_markushgrapher.py
└── eval_real_wavy_hard_eval.py
```

## 工程化重构

### train_moe.py 拆分

原始 `train_moe.py`（4742 行）已拆分为 `src/moe_trainer/` 包下 7 个聚焦模块。
原路径保留为薄 shim（~70 行），兼容既有 shell 脚本和测试。

### 共享基类（class-based）

```
BaseTrainer                         抽象训练循环 + checkpoint
├── BaseCocoMaskRCNNTrainer         COCO Mask R-CNN 共性
│   ├── WavyMaskRCNNTrainer         2 类 wavy 检测
│   └── AttachmentMaskRCNNTrainer   5 类附着点检测
└── WavyUNetTrainer                 二值分割（验证门控）
```

新增 trainer 只需继承 `BaseTrainer`，重写 `build_model`、`build_dataloaders`、
`compute_loss`、`checkpoint_name` 四个方法。

### 统一 CLI

```bash
python -m training.molnextr_markush.cli train moe --epochs 12
python -m training.molnextr_markush.cli train wavy-maskrcnn --epochs 20
python -m training.molnextr_markush.cli calibrate markush-layout predictions.csv
python -m training.molnextr_markush.cli smoke     # 冒烟测试
python -m training.molnextr_markush.cli list      # 列出所有工具
python -m training.molnextr_markush.cli run <tool_name> -- [args]  # 通用分发
```
