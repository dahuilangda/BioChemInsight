# MolSight 小分子与多肽空间解析开发记录

本轮实现保留 MolSight 预训练编码器、字符解码器和原词表，新增图像驱动的空间检测头。没有使用 SELFIES，也没有切换到 MolNexTR 主干。

## 已实现的链路

- 数据：小分子、线性多肽、环肽三类；保持原始 train/validation 化学身份和 component 隔离。重绘坐标直接来自实际 RDKit drawer；多肽沿用已核验的源二维布局，通过带原子映射的立体化学身份检查对应关系。
- 架构：真正的 sin/cos 坐标编码、不同初始采样点的可变形注意力、逐层坐标更新、低秩化学边头。分离普通查询与训练用 DN 查询；推理不读取真实原子数、坐标或排序。
- Loss：正确的 Hungarian 正负类代价差、DN 正负掩码、按化学类别计算的 focal 权重、原子完整状态和可见立体线索监督。对 MolSight 原字符解码器同时进行 complete/peptide 回放与蒸馏。
- 运行：实际反向传播、分模块梯度、参数变化检查、完整 optimizer/scheduler/RNG 检查点和严格重载。导出完整模型及词表，纯图像推理不依赖原始基座 checkpoint。

## 新发现并修复的问题

1. 大型环肽的归一化坐标会碰到 RDKit 立体解析的绝对近零键长阈值。固定乘 100 转为 Molfile 绘图单位后，390 原子训练图以及 383/363 原子验证图通过原始图组装和手性 round-trip。这只是单位转换，不修改预测键、氢或价态。
2. 零学习率预检与“必须发生参数变化”的通用审计冲突。现在分别验证零学习率不变、有学习率真实变化。
3. 导出模型加载误用了训练 RNG 恢复默认值，使未初始化 CUDA 的推理进程报拓扑不一致。推理显式关闭 RNG 恢复；训练续跑仍恢复。
4. 标签 round-trip 不能保证图像清楚。发现一张小分子图最小原子间距 5.47 像素，目视确认局部重叠，已排除。新增严重碰撞门槛：最小间距不低于 768 图上的 8 像素，且不低于中位键长的 1/4。该启发式不能替代完整字体/交叉键/人工质量核验。
5. 同目录续跑现在拒绝日志结尾与恢复 checkpoint 步数不一致，避免重复/倒退更新被混记。

## 真实验证记录

根目录：`/home/dahuilangda/molsight_work/molsight_spatial_v2/`。

- `preflight_v4/state.json`：三次真实零学习率 AdamW 更新通过，三域分别覆盖，最大 390 原子环肽。
- `preflight_v4/parameter_audit_000003.json`：编码器 179、解码器 255 个参数张量均有非零梯度；检测器 179 个张量有梯度，其中 163 个非零。其余为零初始化输出投影暂时阻断的前级梯度，需在有学习率运行中检查恢复，不能宣称每个参数首步均有效更新。
- `preflight_artifact_v4/inference_check.json`：独立导出、加载、图像推理通过，权重保持不变；未训练检测头解析失败，未冒充识别成功。
- `pilot_v6/manifest.json`：97 张训练图（小分子 31 / 线性多肽 32 / 环肽 34），26 张验证图（8 / 8 / 10），包含大环肽。全部是合成开发图，不是真实专利泛化证据。
- `run_v1/state.json`：发现绘图拥挤后，在训练更新开始前停止；实际更新 0，不作为训练结果。
- 回归测试还加入真实空间模型、真实匹配/DN loss 的 CPU 下一步续跑一致性测试。GPU grid_sample backward 存在非确定性，不承诺未来训练轨迹逐字节相同。

扩展执行旧 MolNexTR 数据测试时，当前 MolSight venv 中 Albumentations 2.0.5 缺少旧接口 `safe_rotate_enlarged_img_size`，导致 1 项失败和 11 项 setup error。此问题未混入 MolSight 新链路的通过数，也未通过修改生产依赖来掩盖。

## 尚未证明的能力

当前代码与预检只能证明新链路可以训练、检查和推理，不能证明三域 exact 提升。已有真实 USPTO 图像基线对比见 `molsight_joint_capability_audit_2026-11-12.md`：5100/5607 对 5090/5607 的差值置信区间跨零，两个候选环肽均失败。真实大型环肽测试仍缺乏足够样本。

运行用法见 `training/molsight_spatial/README.md`。后续受控训练以三域图像端到端结果和基座能力保留为准；不把条件 reader 的 28/63 手性结果当成图像解析提升。

## 后续实跑与失败结果（不作为部署成功）

- `run_v2`：三域联合 pilot，250 次有学习率更新后触发 AR 退化保护。AR exact 从小分子 2/8 → 1/8、环肽 1/10 → 0/10；空间 exact 仍为 0。613 个可训练张量全部变化且有限，说明“梯度正常”不能证明能力提升。
- `uspto664_joint250_v2`：固定的 664 组件真实 USPTO 图像，纯图像空间 exact 0/664、parse 76/664。组件级抽样不等于已证明专利家族独立；预训练污染也未知。
- `expanded_v2`：770 train / 98 validation，三域 256/256/258 和 32/32/34，新增大环肽使训练最大 390 原子。全为合成开发图。
- `frozen_expanded_v1`：冻结 MolSight 基座，真实 1000 次更新；空间 exact 三域仍均为 0。原子 F1 小分子 0.774、线性 0.604、环肽 0.618；环肽键 precision 0.0444、F1 0.0789。`final_parameter_statistics.json` 独立比对原始 MolSight 基座，所有 backbone 参数及 buffer 完全相同，optimizer state step 为 1000，模型张量全部有限。
- `uspto664_frozen1000`：空间 exact 0/664、parse 6/664；不因 parse 下降而改用 AR 结果填充空间结果。
- `mining_expanded_v1`：完整载入上一阶段的空间权重，显式重置优化器，单独启用 chemical OHEM ratio=4，再训练 1000 步。累计 prior+stage=2000；原子 F1 小分子 0.843、线性 0.648、环肽 0.687；键 F1 小分子 0.591、线性 0.328、环肽 0.322，环肽键 precision 0.256。追加训练步数、不同增强样本也同时变化，不能把全部差值因果归于 OHEM。分子 exact 仍为 0。
- 原子匹配诊断使用已完成图像推理后的 GT 坐标和元素打分，半径为中位键长的 0.3；它是后验诊断，不是推理 oracle，也不能代替分子 exact。

## 新训练与发布约束

默认冻结基座，仅训练空间头。`--backbone-mode joint` 明确启用联合训练和 complete/peptide AR replay。冻结模式不反复运行完全相同的 AR 解码，而是对全部冻结参数及 buffer 做逐字节审计。

`--spatial-warmstart` 是有记录的阶段切换：保留完整检测头参数，重新建立 optimizer，记录 prior updates；`--resume` 才恢复现有 optimizer/scheduler/RNG/采样游标。实际续跑测试发现 warmstart 来源参数与 resume 的错误互斥判断，已修复。原始 `stereo_mining_v1/update_001000.pth` 保持不变；`stereo_mining_v1_resume_compat/source_migration.json` 记录经过精确源码差分白名单检查的兼容副本。模型、优化器、scheduler、RNG、采样状态、config 等全部非 provenance 字段 digest 一致，只有源码兼容元数据改变。后续仍使用严格完整性检查，不开放任意源码漂移。

Artifact v2 绑定推理必需代码，避免仅修改训练入口就使模型无法推理；模型/词表哈希、实际架构和阈值仍与 checkpoint 交叉核验。修复了配置元组通过 JSON 变为列表造成的假不一致。Topo 评分改为仅移除手性、保留同位素/电荷/所有组分，避免 `isomericSmiles=False` 额外忽略同位素。

尝试 1024/域的数据扩充在 linear train 仅接纳 660 个独立身份后明确失败，未写入 approved manifest。没有用重复身份补足规模。后续改为 600/域；小分子仍有 <=120 原子的开发覆盖上限，不能宣称完整分子所有尺寸都已覆盖。

## 最终边界与实测结果

`stereo_mining_v1` 完成 1000 步后，在兼容副本上严格恢复并完成同阶段 5000 步；加上此前两个阶段，空间头累计 7000 次有学习率更新。没有把预检零学习率步数计入这个累计值。

最终 checkpoint：`stereo_mining_v1_resume_compat/update_005000.pth`。
完整推理 artifact：`artifact_stereo_005000/`。

| 分组 | 开发集完整分子 exact |
| --- | --- |
| 小分子 | 0/32 |
| 线性多肽 | 1/32 |
| 环肽 | 0/34 |

唯一正确的线性样本是 15 原子短二肽 `C[C@H](NC(=O)[C@@H](N)CCC(N)=O)C(=O)O`，不能当成长多肽或大型环肽成功。

`uspto5607_stereo5000/report.json`：完整固定真实 USPTO eligible cohort 的纯图像 exact **3/5607**、topology 3/5607、parse 203/5607；86 个候选 peptide exact 0、2 个候选 cyclic exact 0。三个正确结果分别为 10、11、12 原子的小分子。固定 664 组件代表子集中 exact 0/664、parse 24/664。真实候选标签没有人工确证，不能据此宣称足够的大环肽基准。此新空间模型明显不具备替换既有 MolSight AR 模型的效果。

独立最终参数审计 `final_parameter_statistics.json`：模型和 optimizer 状态均有限；当前 optimizer state step 全部为 5000，prior updates=2000；全部原始 MolSight backbone 参数及 buffer 与基座逐字节相同。正常的数值/参数状态不代表正确识别能力。

位置嵌入复核更正：实际 `MolsightModel` 通过 `register_buffer` 注册 `image_pos_embed`，它并不是可训练参数。因此“实际训练被误分组导致学习率异常”的判断不成立。新增参数分组函数只是未来若将其变为参数时的防护，并对未分类参数 fail-fast；真实联合预检仍为 encoder 179 / decoder 255 / detector 179 个可训练张量，零学习率三域反传和重载通过。

相关 CPU 检查最终为 **142 passed**，包含完整化学状态、同位素保留的 topo 评分、独立 DN、低秩边头、匹配、严重绘图碰撞、难负采样、真实空间模型下一步续跑、参数分组等。旧 MolNexTR 数据测试的依赖冲突仍单独保留，未计为通过。

`scaled_v3/manifest.json` 已接纳 1802 train（600/600/602）、194 validation（64/64/66），并补充源绑定的最大环肽；这份扩充数据**没有用于以上 7000 步实验**。仍需修正 <=120 原子的小分子覆盖限制、检查实际字体/标签重叠、补足真实专利坐标监督及独立人工核验的大环肽测试。当前实际瓶颈仍是全图原子/化学边/立体状态的联合正确率，不能通过隐藏失败、事后修价态或拼接 AR 结果来宣布解决。

最终同一 98 图开发集的后验定位诊断（不是 molecular exact）：

| 域 | 原子 F1 | 化学键 F1 | 原子数 MAE | 原子数完全正确 |
| --- | --- | --- | --- | --- |
| 小分子 | 0.9254 | 0.7334 | 2.25 | 5/32 |
| 线性多肽 | 0.6752 | 0.3469 | 9.375 | 2/32 |
| 环肽 | 0.7336 | 0.3527 | 13.0 | 1/34 |

98 图仅 8 图预测原子数正确，因此当前明显受完整原子集合和假边约束；匹配上的原子 charge/isotope 等局部状态正确率很高，也不能弥补缺失或重复原子。诊断原始报告位于 `diagnostics_stereo_005000/report.json`，比较图为同根目录 `development_comparison.png`。机器可读最终汇总为仓库 `output/ocsr_audit_2026-11-12/spatial_development_summary.json`。
