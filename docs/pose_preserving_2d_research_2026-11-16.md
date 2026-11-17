# 保留来源姿态的二维结构清理：调研与实现

本次处理的是二维文献图中的排版姿态，不是三维结合构象。“保持姿态”包括来源原子对应、左右关系、整体方向、环骨架和立体化学；局部纠正键长、键角会改变少量坐标。因此，整体重排、再旋转对齐，不作为保姿态方案。若要求所有坐标完全固定，则不能同时改变几何。

## 现成能力与源码核查

| 方案 | 可复用能力 | 与本任务的关系 | 验证方式 |
|---|---|---|---|
| RDKit constrained depiction | `coordMap` 固定对应原子；`GenerateDepictionMatching2DStructure` 支持硬约束与 align-only | 适合把正确分子图接到来源母核上；align-only 不会自行消除局部畸变 | 阅读官方文档与源码，并实测 |
| Schrödinger CoordGen / RDKit wrapper | 模板、固定/柔性坐标约束、键/角/碰撞项 | 默认生成与柔性约束不能作为严格位移保证；全固定坐标也无法充分清理几何 | 阅读原生与 wrapper 源码，并实测 |
| CDK StructureDiagramGenerator / LayoutRefiner | 固定原子、固定键；rotate/bend/stretch overlap refinement | 默认路径包括反射、反转等操作；需要固定集合和额外姿态验收 | API 与源码核查；未声称进行完整 CDK 运行时对照 |
| OpenChemLib CoordinateInventor | KEEP/PREFER marked atom coordinates；模板和局部生成 | KEEP 适用于保持来源核心；PREFER 仍允许改变坐标，不能代替严格上限 | API 与源码核查；未声称进行完整 OCL 运行时对照 |

主要来源：
- RDKit 官方二维坐标与约束参数：https://www.rdkit.org/docs/source/rdkit.Chem.rdDepictor.html
- RDKit CoordGen 参数：https://www.rdkit.org/docs/source/rdkit.Chem.rdCoordGen.html
- RDKit CoordGen wrapper 源码：https://github.com/rdkit/rdkit/blob/master/External/CoordGen/CoordGen.h
- CoordGen 原生 minimizer：https://github.com/schrodinger/coordgenlibs/blob/master/CoordgenMinimizer.cpp
- CDK overlap refiner：https://github.com/cdk/cdk/blob/main/tool/sdg/src/main/java/org/openscience/cdk/layout/LayoutRefiner.java
- CDK 固定坐标 API：https://cdk.github.io/cdk/1.5/docs/api/org/openscience/cdk/layout/StructureDiagramGenerator.html
- OCL CoordinateInventor：https://github.com/Actelion/openchemlib/blob/master/src/main/java/com/actelion/research/chem/coords/CoordinateInventor.java

结论不是“找到了一个所有情况最好的 Clean 2D”。现成引擎提供了可靠的约束排版原语，但本产品的目标函数包含来源忠实度，需要另加明确的位移、方向、拓扑和立体化学门控。本次实现复用 RDKit 的分子图、原子映射、约束生成和绘图；用一个有界局部清理器处理已有坐标。没有引入新的化学识别权重，也没有声称提出通用 SOTA 排版算法。

## 截图问题的原因与修复

5.png 中长分子的 MOLBLOCK 为空。来源任务的旧记录也没有原始坐标，因此缩略图只能从 SMILES 生成新的标准排版，产生了左右关系改变。已重新取得该来源图的坐标预测，将正确分子图与对应元素/连通性映射，经过完整立体化学往返检查后恢复坐标。坐标预测中的错误键级和额外原子没有被当作正确化学图采用。

6.png 中的母核曾经由完整名称共识重建后直接生成标准坐标，丢失了来源母核方向。现在共识母核会先映射到来源母核坐标，再组装片段。17 个已接受成员和上述长分子共 18 条记录更新了坐标；所有更新均检查了 canonical isomeric SMILES 一致。未通过原有审核的成员没有因此转为已接受。

识别流程在 MolSight 替换 keeper 化学图时，会尝试保留 keeper 的几何，即使 keeper 化学解码被拒绝。手动框选解析也会在识别阶段获取来源坐标；缩略图和打开编辑器时不再次调用识别模型。无法可靠映射时记录 `source_coordinates_unavailable`，界面明确提示比较来源，不假称保留了不存在的 pose。

## 有界局部清理

归一化后的目标包含键长、局部角度、原子/键避碰和对来源位置的约束。使用解析梯度、L-BFGS-B、最多 48 步；空间筛选减少远距离原子对和原子/键对的计算。每个坐标分量相对来源的移动范围为 0.32 个来源中位键长，接受后的欧氏移动还需满足 0.32√2 上限。

局部结果必须通过：
1. 不产生新的键交叉；
2. 保留来源局部有向面积的符号，避免局部翻面；
3. 使用 det=+1 的刚体校正清除求解产生的细小整体漂移，禁止反射；
4. 位移预算和质量目标检查；必要时回溯减小局部步长；
5. Molfile 写入/读取后，完整 isomeric SMILES 不变。

大于八元的环骨架采用固定坐标。环肽不套用小环正多边形，也不按普通支链角度规则拉动主环；只清理可移动外围。若坐标坍缩、非有限、不是平面输入、超出规模预算，或不能通过验证，则保留来源并返回可识别状态，而非强行全量重画。

来源图到正确图的映射优先精确子结构，再使用有时间上限的公共核心；元素、同位素、电荷均参与约束。对于几乎完整但漏了一条宏环闭合边的几何图，允许受严格覆盖率限制的闭环对应，并检查新增边长度和立体化学。该步骤仅转移坐标，不采用几何解码器的错误化学图。

列表和 JSME 使用同一清理流程。SMILES-only 的合法文字来源结构会由后端统一准备坐标，避免两个绘图引擎各自生成不同布局。

## 同机对照

环境：RDKit 2026.03.5，SciPy 1.16.0。冻结输入为修复前 SR 任务中可读的 28 个 Molfile，以及 6 个带固定扰动的合成输入，合计 34 个。截图长分子因修复前没有坐标，作为独立的来源坐标恢复案例，不混入这 34 个输入。

| 方法 | 中位耗时 ms | 中位姿态 RMS / 键长 | 最大样本 RMS / 键长 | 整体方向变化 >30° |
|---|---:|---:|---:|---:|
| RDKit 全量重新生成 | 0.11 | 1.639 | 9.701 | 18/34 |
| RDKit align-only | 0.02 | 0 | 0 | 0/34 |
| CoordGen 全坐标柔性约束 | 1.42 | 0.237 | 0.903 | 0/34 |
| CoordGen 全坐标固定 | 0.30 | 约 0 | 0.001 | 0/34 |
| 有界局部清理 | 6.55 | 0.073 | 0.307 | 0/34 |

RMS 用同序原子、居中后各自按中位键长归一化计算，不允许通过旋转/反射降低对照 RMS；这与求解器用固定“来源键长”定义的位移上限是不同口径。有界方法的中位键长变异系数约 0.0066，固定来源约 0.0158；P95 耗时约 7.6 ms。34 个输出均通过立体化学往返检查。固定/align-only 方法位移小，但几何基本保持原样。

耗时只代表排版阶段，不包括图片识别、加载模型、首次导入 SciPy、PDF 解码或绘图。来源长分子的辅助坐标预测约 1.16 s，不含模型加载；该操作发生在识别/恢复阶段，不是每次显示。

合成环肽压力测试：8、20、40 残基，分别 42、105、210 个原子。主环原子均固定；20、40 残基清理约 9、26 ms。第一次调用还包含约 0.1 s 的库加载。这里验证的是排版约束与立体化学，不是 MolSight 在真实环肽上的识别准确率。

两篇文献和少量合成结构不能证明对任意分子的统计泛化能力。当前正确性保证是输入图不被改写、已取得的来源姿态有明确约束、失败可追踪；仍需独立的大规模排版基准衡量主观美观与各种复杂结构的覆盖率。

## 证据与复现

`output/pose_research_20261116/` 保存了原始记录快照、来源坐标恢复、组装提案、引擎对照明细、环肽压力测试、更新清单及前后对比图。任务保存了可撤销的数据版本。浏览器截图分别验证长分子缩略图、长分子 JSME 和组装成员 JSME。

测试覆盖来源方向、环几何改善、局部位移预算、立体化学和 E/Z 往返、同位素映射、宏环骨架固定、漏闭合边恢复、退化坐标拒绝，以及手动框选保留来源坐标。
