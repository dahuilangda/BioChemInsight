# MolSight 小分子与多肽同步提升：独立审计与继续执行条件

日期：2026-11-12。用户目标是完整小分子与多肽同步提升，重点环肽；暂不处理 Markush/fragment。不使用 SELFIES，不从头训练基础模型。本文是实际源码、数据、参数和推理复核，不能以数值有限或单域 oracle 成绩代替产品验收。

## 结论

旧空间图路线有可复现的条件学习收益，但尚无本次材料支持的图像端到端通用识别改进。当前队列含数据错位、错误 loss、错误匹配与不可执行脚本，不能直接启动噪声训练或扩展100k训练。原始代码、日志和权重保留，审计未修改这些文件。

审计目录：`/data/BioChemInsight/output/ocsr_audit_2026-11-12/`。`inventory.json` 绑定107个原始文件，其中93个代码/报告/日志有副本，14个检查点有SHA256。三个独立审查分别覆盖数据、架构/评估、训练/参数；主审另做静态检查、评估契约探针和实际GPU复现。

## 1. 哪些成绩是真的，分别证明什么

### 1.1 原始空间 reader：GPU独立复现

使用冻结源码副本和原始权重，63条唯一 record_key、没有优化器更新。重新得到：

| 检查点 | parsed | topology exact | 完整带手性 exact | 楔键头来源 |
|---|---:|---:|---:|---|
| wedge_joint | 60/63 | 19/63 | 19/63 | 自身 |
| phase1_last | 61/63 | 29/63 | 28/63 | wedge_joint |
| focal_last | 61/63 | 25/63 | 24/63 | 自身 |

证据：`oracle_reproduction/report.json`、`predictions.jsonl`、`run.log`，含源代码、视觉基座和reader权重哈希及预测SMILES。

**性质必须标为 oracle-coordinate conditional reader diagnostic**：`canonical_eval.py:141–172` 从验证标签/原始绘图获得真实原子数、词法顺序和坐标。模型预测元素/键/楔键，但没有自己从图像发现全部原子。因此这些数值不能与无这些辅助输入的 AR SMILES 1/63 当成公平端到端提升。旧脚本`canonical_eval.py:219`还把`phase1_last.pth`命名为`focal_last`输出；本次以实际文件与SHA纠正。

63张原始validation具有63个独立component，与适配train的component和literal SMILES重叠均为0。这一事实与内部heldout泄漏要严格区分。

### 1.2 独立 reboot 支路

`training/ocsr_reboot/model.py` 是 **MolNexTR warm-start**，不是原MolSight参数微调。当前118项CPU测试通过；v3实际10000步日志、参数及优化器审计有效。但最终自由生成只有train **15/16**、validation **0/5** exact。它证明预训练模型可记忆大部分小诊断集，未证明泛化，更不能记作MolSight能力提升。

旧执行日志`docs/ocsr_reboot_execution_2026-11-11.md`中“正在训练”的段落已过时：现场无这些训练进程，新v3状态是`training_finished_requires_acceptance`。旁支`free_gen_eval.py`跳过正式预处理、混合train/val且未约束EOS，不能覆盖正式`train.evaluate`结果。

## 2. 必须先处理的缺陷

| 优先级 | 证据位置（旧空间图目录） | 实际问题与影响 |
|---|---|---|
| P0 | `stage5d_detect.py:63–65` | Hungarian focal分类代价写成positive+negative，而非positive−negative。同坐标下p=.5比p=.9代价更小，匹配方向错误。 |
| P0 | `stage5d_detect.py:308–310` | DN标签用batch维构造长度1，再广播。实际64个DN查询全部被标正，包括32个随机负查询。CPU梯度证明负查询也被推动为正。 |
| P0 | `stage6h_focal.py:76`及i/l/m复制函数 | `grouped[...,0].softmax(-1)`沿候选原子j轴，而不是五种化学类别轴。200原子容易负样本权重实际约0.990025，正确类别概率应约0.000004。γ比较和“难负focal有效”的机制归因无效；已有分数仍是实际条件分数。 |
| P0 | `convert_mass_gen.py:21,31–45` | 只遍历前6000个MOL；标签使用原始MOL坐标缩放/clip，图像却由RDKit独立绘制。64个MOL共18499原子独立对照平均误差559.85px/1024，68.43%原子压在x=0或y=0。尚未发现mass_gen_cache，不能断言旧权重用过这些错误标签，但不能让它进入下一轮。 |
| P0 | `stage6m_continue.py:281–284,382–383`；h/l类似 | 加载楔键头后又新建同名头，覆盖已加载权重。继续训练/γ消融混入头部重新初始化，不能只归因共享trunk互扰。旧检查点不绑定完整源码，历史执行范围另受限制。 |
| P0 | `merge_route*.py`；v3:112–115 | 合并路径仍用真值gold_n选top-k；v3又使用未定义top_indices，且wedge logits计算后未应用。不是有效的完整手性图像端到端路线。 |
| P0 | `stage7_noise_finetune.py`、`noise_finetune_v3.py` | 静态检查分别发现7处、2处未定义名称；另有GPU/CPU坐标混用、首步评估q/r不存在、标为focal实际调用普通loss等路径。现有日志有真实设备错误退出。不能直接启动。 |
| P1 | `stage5d_detect.py:69,80–86`；h/i切分 | 图片级随机heldout：stage5d重建30条中24条component已见于train；h/i为5/30。内部F1不能当独立化学实体泛化。 |
| P1 | `spatial_atom_graph2.py:39–48` | CoordFourierEncoder没有sin/cos，只把x/y乘频率；8维列输入的实测rank为2，与文档声称的Fourier编码不符。修改基础函数会改变检查点语义，必须版本化，不能偷偷改后仍沿用旧锚点解释。 |
| P1 | `spatial_atom_graph2.py:61–67` | DCA四个采样点的零初始化保持对称；CPU连续三步点间spread为0、采样权重梯度为0，名义四点退化为重复采样。真实旧权重也有对应不动参数。 |
| P1 | `patent_augmentation.py:52,70,98–112` | JPEG传不支持的quality_range、RGB直接OTSU报错后均静默跳过；intensity未用，训练目录中没有调用证据。代码存在不等于增强已生效。 |
| P1 | `canonical_eval.py:173–185,212–240` | cache静默跳过；anchor不校验n/keys；失败return导致exit0且旧JSON仍在；chord_fp/missing从未计算却默认输出0。不能称“评估纪律已完备”。 |
| P1 | `stage6i_twophase.py` phase2结尾 | phase2只存JSON，没有最终完整权重；phase1_last不能代表两阶段最终模型。 |

证据分别见`architecture/{findings.md,probe.py,probe.json}`、`parameters/{findings.md,parameter_audit.json,cpu_training_math_probes.json}`、`data/*.json`、`evaluation/probe.json`。静态F821总计17处，完整输出在`undefined_names.json`。

未误报的项目：三份single logits再做概率合并在现有化学loss/评估下自洽，相当于single加log(3)先验，不是单凭重复代码就判错；DN常规查询与DN查询是两次独立执行，并未证实“拼接无mask造成推理真值泄漏”。

## 3. 梯度与最终参数的结论边界

五个重点旧检查点（phase1_last/focal_last/wedge_joint/backup_last/detect2）均为有限FP32，无NaN/Inf，最大绝对值约5.16–6.51；没有发现数值爆炸。但都缺optimizer、scheduler、RNG及真实step，不能据此证明训练全过程梯度正确或精确续训。detect2配置标80000，现有日志最后60000；100k完成无法证实。

phase1的坐标编码器、edge scorer和部分trunk确实变化；feature_proj/element_head不变符合部分主动冻结，context_proj在run_reader中不调用。不能把全部不动参数一律判为梯度故障。DCA采样权重及部分coord_head长期不动则与已复现的结构问题对应。

reboot v3的10000条连续日志中，encoder/decoder/graph_head梯度范数逐步有限、非零，无missing梯度；最终494个优化器参数的step均为10000，形状/名称覆盖、二阶矩、payload与源码哈希通过。**正确更新与独立泛化是两个要求，0/5 validation仍不通过产品验收。**

## 4. 数据分布：分层均衡，不能只扩环肽或机械均匀

18,883个accepted生成分子确实是独立isomeric SMILES，另10个生成失败；但全部属于`head_to_tail_patent_large`、189–403原子、全部training_admitted=False。不存在已准入且对齐的28534张通用训练图证据。

后续数据契约应包括：

1. 主域分别覆盖完整非肽小分子、线性多肽、环肽；环肽再分头尾闭环、侧链闭环/套索、二硫键、多重闭环及非天然修饰。不用原子数代替化学类别；完整盐/多组分结构也不能简单当fragment排除。
2. 域内按原子数/图上键长、环大小、元素/电荷/同位素、立体中心、来源和绘图风格分层。稀有困难层补样，避免同一分子的多渲染图冒充更多化学多样性。
3. 以真实目标场景分布保留自然权重，同时记录每域/每层实际优化器更新曝光；在开发集上校准采样比例，不能先指定一个“最优均匀比例”。
4. 按完整分子身份/component分组，能取得专利/文献family时进一步按family划分；先冻结划分再渲染/增强。当前无法验证历史预训练重叠，不宣称彻底未见。
5. 原子标签需覆盖真实小分子的元素、形式电荷、同位素与完整立体语义；旧elements=36及仅元素头不能作为通用完整小分子表示。

## 5. 专利化增强必须可见、同步、可审计

- 先从真实专利图测量分辨率、键/字线宽、压缩、灰度、倾斜和噪声分布，生成匹配范围的混合增强，保留一定比例清晰图。
- 缩放、旋转、裁剪、padding用同一变换同步图像与坐标；坐标来自实际drawer.GetDrawCoords及之后的真实输入变换，而非原始MOL几何。完整结构不得被裁掉。
- JPEG、模糊、二值化等不能修改分子标签，也不能把字/楔键损坏到不可辨认后继续按完整正确图监督。
- 保存分层增强接触表、参数/种子、图像SHA及变换链。增强异常应显式计数/失败，不得吞异常后仍称增强成功。
- 分别做无增强/真实范围增强的受控消融；模型、数据划分、预算和初始化都保持一致。仅噪声teacher坐标不能模拟检测器漏原子、重复原子和元素错误，σ要按实际定位误差/键长尺度校准。

## 6. 真实专利评估与共同验收门槛

官方[Keylab/OCSR-Benchmarks](https://huggingface.co/datasets/Keylab/OCSR-Benchmarks)区分USPTO真实5719张和Indigo/ChemDraw合成图，不能混算。本地真实原图和MOL已完成固定版本核验：官方revision为`2fdcdc59067acb36d84ddc82c1ace24c91d5ad71`，5719张逐像素一致；86条官方标签无法严格解析，另6条可解析标签与MOL冲突、7条MOL无法核验，重复结构也单列。最终严格可评分集合5607张/5607化学身份，只有664个来源components。对当前适配train+validation的已可比较身份/图像/component，重叠为0；扩展标签完整身份及历史预训练曝光仍未证明。另导出664个component各一图子集。结果以独立目录`/home/dahuilangda/molsight_work/ocsr_real_patent_eval_v1/`的`benchmark_cohorts.json`和`additional_cohorts_v1.json`为准。适配数据去重不等于证明原始预训练没有见过公共benchmark。

真实测试要求：

- 输入仅图像；原子数、坐标、类型与完整结构均来自模型，不得gold_n top-k或teacher坐标。
- 真实小分子、真实线肽、真实环肽分别给分母、解析率、无立体topology exact、带完整立体RDKit graph exact；另列尺寸/风格/来源困难层。USPTO5719不足以单独验收环肽。
- 正式测试保留自然分布，不把所有层机械均匀化；另设均衡诊断集和困难集。所有失败保留在分母，不能因缺标签支持或解码失败静默drop。
- 与原MolSight及公平预算的MolNexTR在同图、同化学等价契约下做配对比较。预先声明是否使用修复/组装规则，并单列原始预测与后处理后的结果。
- 小分子与环肽必须各自有独立端到端收益，多肽总体收益及线肽保留也要成立；禁止用加权总分掩盖单域退化。数据量不足时报告不确定性，不能以少数样本的涨跌宣称通用提升。

## 7. 下一轮执行顺序

1. 先冻结现有资产与本次条件复现；不再覆盖同名last文件。
2. 在独立版本中合并重复训练/评估代码，修正匹配符号、DN形状、focal类别轴、头重复初始化；加入真实函数的梯度方向、形状、概率平移不变性与初始化身份回归检查。
3. 修复并独立准入图像/坐标生成，落实三域component级划分、真实范围增强和实际曝光审计。
4. 继续以MolSight为目标基座做受控小预算真实更新，验证每模块梯度、参数变化、完整checkpoint、精确resume；另一MolNexTR支路只作明确命名的对照，不能混报成MolSight提升。
5. 先验证完全不依赖真值坐标/数量的合并路线，再考虑扩大步数或强化学习；先验收三域真实图指标，再进入真实服务、资源与前端联调。不要为“首个非零”牺牲输入纯净和验收范围。

当前状态：审计发现已具备复现证据，旧队列未启动，整体产品目标仍未达成。

## 8. 本轮已启动的真实图基线（不是训练）

`/home/dahuilangda/molsight_work/ocsr_real_patent_eval_v1/molsight_baseline_comparison_v1/plan.json`冻结MolSight SFT2000与graph500两个既有检查点，在相同5607张真实图、FP32、512输入、2048长度预算下分别推理，优化器更新为0。现已完成：SFT2000为5090/5607（90.779%）带手性exact，graph500为5100/5607（90.958%），增幅0.178个百分点；664个component聚类bootstrap的95%变化区间为[-0.205,+0.563]个百分点，跨零，不能认定可靠改进。拓扑exact为5287→5308，两个未人工复核的环肽候选均为0/2。`comparison.json`同时保留终止且exact与尺寸层诊断，不把5607视图当独立试验。

该真实集严格子集只有86个未经人工核验的肽候选、2个环肽候选（36及107原子）、没有189原子以上的大型环肽，不能替代真实环肽验收。这些真实图成绩不构成已解决目标大环肽的证据；这些基线也不是修正上述P0后的新训练模型。所有计划、源代码、checkpoint、数据与预测哈希会留档，无自动部署。

## 9. 下一步已开始的修正（2026-11-12）

新增`training/molsight_multitask/focal.py`，统一类别轴概率、固定负样本分母、逐图平衡与显式缺失标签；明确区分detach权重的surrogate和完整focal梯度。`test_chemical_focal.py`六项检查通过，包括候选数量不变性、共同类别logit平移不变性、γ=0与既有目标及梯度一致、空监督零梯度、非法输入拒绝。尚未接入任何训练入口，不宣称修正后模型已有收益。

Matching及DN复用已经通过审查的`training/ocsr_reboot/objectives.py`纯张量数学函数，它们与MolNexTR模型无依赖；下一版MolSight检测训练必须显式接入并验证，不能因helper存在就标记训练正确。旧检查点若要精确重启需使用其冻结源码，本次新loss文件不应被悄悄并入旧resume契约。

## 10. 部署升级：peptide_visible + multiscale 上线（2026-11-12，ZCode 会话）

在保持审计结论（产品目标未达成、验收双门制）不变的前提下，把已实测更优的已有资产部署到前端服务：

| 项 | 旧部署 | 新部署 |
|---|---|---|
| artifact | artifact_visible_final_v1 | artifact_root_hard_v1（backbone=graph500 冻结，sha256 9208dbd0…） |
| 分支策略 | preserve_ar（硬编码） | peptide_visible（MOLSIGHT_POLICY 环境变量） |
| visual 分辨率 | 单尺度 768 | multiscale [768,1024,512]（MOLSIGHT_GRAPH_SIZES） |
| 推理校准 | calibration_final_v1（绑定 visible_final） | 移除（multiscale 验收即用 artifact 默认 threshold 0.5） |

预期成绩（全部为 root_acceptance_v1 已实测数字）：USPTO 5607 保持 5102（AR 分支逐字节不变）；dev 线肽 33→58/64；dev 环肽 32→57/66；dev 小分子 33/64 不变；真实 macro 3→4/15。

端到端验证：部署容器内 worker ready 事件（policy=peptide_visible, threshold=0.5, graph_sizes=[768,1024,512]）；环肽样本走 visible_graph 分支 exact；小分子走 molsight_ar 分支。冒烟脚本与记录：`/home/dahuilangda/molsight_work/deploy_upgrade_v1/`（smoke_test.py、DEPLOY_RECORD.md）。回退：revert worker 与 compose.override 两文件 + `docker compose up -d`。

补充评估：artifact_root_diverse_v1 在 dev 194 multiscale 为 145/194（hard 146/194），不改变 hard 部署决策；结果在 `validation_194_multiscale_diverse_v1/`。

已确认的下一训练项根源：visual head 训练集（diverse_small_v2/manifest，9316 条）全部合成渲染、0 真实专利图——这是 visual 分支真实图仅 1611-1926/5607 的直接原因。真实数据混训需引入 USPTO 评测集之外的真实专利图训练源（禁止用 5607 评测集训练，避免烧毁基准），置于 P0 修复完成之后。

## 11. P0 修复完成 + macro15 归因 + AR 候选解码实验（2026-11-12 深夜，ZCode 会话）

### 11.1 P0 修复（CPU-only，33/33 回归通过，零 checkpoint 触碰）

- stage5d_detect.py：Hungarian 代价改 positive−negative（p=0.9 代价 −1.3986 < p=0.5 −0.0866，方向已证反转修复）；DN 标签改逐查询 64 维（负查询梯度 +0.0035 推离正类，旧 bug −0.0012 推向正类）
- stage6h/l/m：focal 概率轴修复为类别轴（委托已过审 focal.py；易负样本权重 0.990025→3.4e-11；γ=0 与 CE 梯度差=0）
- stage6h/l/m：楔键头二次构造覆盖已删（加载后逐参数 torch.equal 校验，fail-closed）
- canonical_eval.py：anchor 失败 exit=2、cache 三重哈希校验、chord_fp/missing 在 valence-greedy 最终图真实统计
- **chord_fp 基数定论**：wedge_joint 最终图净键盈余 +1.367 键/分子（phase1 +0.82、focal +0.92）；1137 是早期 argmax 时代口径、137 是 argmax 图口径、0 是从未计算的占位——三个历史数字互不可比

### 11.2 真实大环肽 15 例归因（与 root_acceptance 同口径）

- AR 3/15、visual hard 2/15、visual diverse 3/15、路由 hard 4/15 / diverse 5/15
- **路由零损失**：hard/diverse 均已打满各自 oracle，AR_correct_overridden 全空
- 失败分类：残基漂移/重复 7、环闭合接位错 4、解析失败 4（内容先漂移）、单手性 1、visual 装配/价态层 11（2 例元素组成全对仅键级错）
- **9 例可在解码/装配层修复，4 例架构性**（95-137 原子超宽拉链布局 + 重复退化）
- 图像侧：分辨率全部充足（512px 反而最稳）、零缩写障碍

### 11.3 AR 候选解码实验（部署同款 backbone，无重训）

- **发现并修复运行时隐藏缺陷**：BeamSearchDecoder 重排 beam 时不重排 KV cache（`rearrange_kv_cache` 被注释），beam 路径从未可用；cache-aware 修复版在 `deploy_upgrade_v1/beam_decoder_v2.py`
- macro15 候选 oracle：greedy 3 → ∪beam4 4 → ∪beam8 5（#13 金标在 beam8 rank5）；#11 金标不在任何候选（架构性）
- 候选选择两策略（oracle-free）：conservative 3/15（greedy 总能解析时不增益）、aggressive 4/15（化学打分选中 #13；结合部署路由 visual-first 规则预期 6/15）
- 待 dev 肽集回归门结果（评估运行中）；非肽图像保持 greedy 逐字节不变，USPTO 无回归风险

### 11.4 候选选择部署决策（最终）

dev 肽集 130 例 + macro15 全量评估：组合路由（visual-first-peptide 外壳 + AR 候选内核）
- conservative：零回归零增益（consensus 增益全部被 visual-first 覆盖）
- aggressive：macro 4→5（+#0 +#13、−#8 回归），dev 115→114（−#25 回归、零新增）
- **结论：不满足"dev 不回退"部署门，不部署。** beam 候选的增益（金标在候选集中）真实存在，但 oracle-free 打分规则尚不能无回归地兑现；CacheAwareBeamSearchDecoder 保留为评测/训练资产（运行时 beam 的 KV-cache 缺陷修复）。

## 12. pixel 语料控制组训练重启（2026-11-13 凌晨）

- full_visible_groups_v3 语料 56,096 条全部通过图像尺寸/坐标边界静态扫描（0 坏记录）——其失败根因是**语义级坐标框架错位**（bounds 合法但位置错，即 convert_mass_gen P0），静态扫描无法检出；pipeline.json 已指定 pixel_aligned_corpus_v1（56,094 条，training_release_approved=True）为替代语料
- 启动过程中的两个守卫按设计拦截并修正：(1) 训练源码溯源绑定要求 CWD=仓库根（launcher 加 os.chdir）；(2) **warmstart inference_sources 检测到 inputs.py 被修改即拒绝**（本次曾尝试加边界舍入钳制，与 warmstart 契约冲突——已从 source_snapshot 恢复原版字节，钳制未引入；若 pixel 语料训练再触发边界崩溃，须版本化新模块而非改旧锚点）
- control 训练配方与失败运行逐字相同（visible、native 预处理、train_sizes 512/768/1024、warmstart diverse_hard_1500、20k 预算/5k 评审边界、seed 46、lr 5e-5），GPU1 顺序执行
- 早期健康指标：loss 单调降（428 原子环肽 1.79→0.15）、梯度各组有限非零、无 OOM（~2.8GB/16GB）

## 13. pixel 语料控制组训练结果与晋级决策（2026-11-13）

**实际预算**：计划 20k，实际 16k（update ~16k 时命中 native 白裁剪浮点舍入把边界原子映射到恰好 w 的 `Source coordinates out of bounds`——该问题需版本化新模块解决，不能为续训偷改被 warmstart/resume 溯源绑定冻结的共享源码；内部评估 6k-16k 已平台期，边际价值低，如实按 16k 结算）。

**验收结果**（同一 multiscale 协议 + cached AR 路由，dev194 分域）：

| 域 | 部署 root_hard | 5k pixel | 16k pixel |
|---|---|---|---|
| 环肽 66 | 57 | 56 | **58** (+1) |
| 线肽 64 | 58 | 55 (−3) | 55 (−3) |
| 小分子 64 | 33 | 33 | 32 (−1) |
| 194 总 | 148 | 144 | 145 |
| macro15 路由 | 4/15 | **5/15** | 4/15 |
| USPTO 150 抽样路由 | 131（冻结 AR） | 未测 | **132**（+1，无回归） |

**晋级决策：不晋级，保持 root_hard 部署。** 两个候选都不满足"单域不回退"门（16k 丢线肽 3+小分子 1；5k 丢环肽+线肽）。USPTO 侧两候选均无回归（AR backbone 冻结一致）。

**结论与教训**：
1. 56k 语料（39k 小分子）未转化为肽域收益——采样配比（40/20/40）下线肽层仅 3,231 条记录被稀释，线肽 −3 的回归与容量转移一致
2. macro15 上 4-5/15 的波动在 n=15 噪声内；5k 检查点曾达 5/15，但不足以单独晋级
3. 下一轮前置条件：线性层过采样/配比重平衡 + seed/预算敏感性（verified 对照臂）+ 边界舍入的版本化修复（新模块，保持旧溯源锚点冻结）
4. 本轮全部产物（检查点、验收报告、日志）保留于 pixel_corpus_run_v1/，部署服务未被触碰

## 14. 前端端到端浏览器走查（2026-11-13，Playwright 真实 UI 流程）

测试：scirep_benzimidazole.pdf（Scientific Reports 苯并咪唑 SAR 论文，14 页，GitHub 仓库无 data PDF，用项目本地数据 PDF）。

### 发现并已临时修复（容器层，重建即失）
1. **web 501**：容器内 static-server 为 8 月旧版（无 do_POST 代理），仓库新版未进容器（docker/ 不在挂载清单）——docker cp 新版 + restart 已修；持久化需镜像重建或把 docker/static-server.py 加入挂载
2. **worker 缺 ultralytics**（清华镜像补装 8.4.199）：MolDetv2 是唯一结构检测器，缺失=结构提取全瘫。此前 WO2024180521 "无结构" 假象即此问题——已列入修复后重测

### 结构提取链路实测（修复检测器后）
- 自动页面检测正常：LLM 选中 [2,3,4,8]；检测器工作；box gate + MolNexTR + 识别链全通
- **strict/balanced 均 0 结果**：11/10 个候选全被丢——MolNexTR 识别全部成功（含活性先导 G 的 SMILES 完全正确），但 visual-role 分类器判 G=reactant、表格结构=table_entry，而 "only target/final-product kept" 过滤把非 final 全丢
- **permissive 仅 2 条**：6a-r 系列母核（SMILES 正确）+ 表格 BR 结构；reactant/intermediate 在 permissive 仍被丢
- page 8 段分类返回空字符串被静默跳过
- 一个单结构框被 box gate 误判 multiple_structures（VLM 描述了整页 Figure 1 而非框内内容）

### role 分类器根因方向（待 agent 深度调研确认）
分类体系缺"无反应上下文的独立分子"角色；prompt 无"无箭头时禁止 reactant/intermediate"约束；role 被用作删除依据而非元数据。

### UX 其余观察
- Jobs 模态正常（状态/消息/进度清晰）；主视图 12 分钟管线期间进度反馈不足
- "Markush & fragment assembly: Unavailable" 无解释
- 任务 completed 但 0 结构时无漏斗诊断（检测→gate→识别→过滤 各步计数）

## 15. visual-role 分类器根修 + 深度 code review + 前端修复（2026-11-13）

四路独立 agent 调研（role 根因 / UX / code review / 文档规范）后的集中实施。

### 根因（第五个 bug 为全场放大器）
- **bbox 坐标转置**：`bbox_yxyx_to_xyxy` 对 XYXY 框做 x/y 交换 → 红框画在错误位置（非近方形框基本随机）→ box gate 与 role 判定的 VLM 输入全被误导，幻觉（"joined by a plus sign"）由此诱发
- **归一化层销毁合法枚举**：classify 契约允许 noise/uncertain，但 `_STRUCTURE_TYPE_SYNONYMS` 不含它们 → 归一化为空串 → 静默丢弃（page 8 段、Figure 1 一半化合物由此蒸发）
- **role 分类体系缺"无反应上下文"类**：无箭头时模型只能硬套 reactant（G 被误判）
- **gate 输入双页拼接 + 红框错位**：2382px 拼图里红框占 0.3%，VLM 只能对整页讲故事
- **role 被用作删除依据且与 strictness 无关**：strict/balanced 全丢 reactant/intermediate/table_entry

### 已实施修复（全部卷挂载即时生效）
- A 坐标统一 XYXY（image_utils + structure_parser 两处定义 + scheme review 解包）
- B 归一化承认 noise/uncertain；静默丢弃段改为 filtered 留痕行；classify 三处调用补 audit_path
- C VISUAL_ROLE 新增 `standalone_compound`（schema+prompt+examples+scheme prompt 的 context gate 与反幻觉条款）
- D gate 输入改单页图（双页拼图另存 crossref）；box gate prompt 加"只看框内"契约
- E(简化) gate `uncertain` 在 balanced/permissive 放行；`partial_or_truncated` 先补框重试（pad 20%）
- F 过滤语义重设计：`role_filter_should_drop(role, id, strictness)`——permissive 不删 role；balanced 只删明确反应位；strict 加 ID 要求；空 SMILES 行落 filtered
- G（观测）段级丢弃全部留痕（原因+计数进 filtered CSV）

### code review 修复（同批）
- P0：fragment 救援分支永假比较（quality_reason 前缀）；`_constants` 未定义致 label-rewrite 静默 no-op（2 处）
- P1：proxy 装饰器 try/finally；CSV 公式注入消毒（`_csv_safe_frame`，6 写点）+ `read_csv(dtype=str)`；子进程异常兜底 task→failed；imread None / 缺页计入 failed_pages；P1-2 页级 review 一致性（agent 建议，后批实施）
- 死代码：batch_process_structure_ids、cost_time、predict_segment_file/_predict_molnextr/health_check、未用 import

### 部署持久化
- compose 挂载 docker/static-server.py + start-web.sh（防镜像陈旧）；Dockerfile 钉 `ultralytics==8.4.199` + 导入冒烟；镜像已重建、容器已重建
- 教训两条：①单文件 bind mount 的 inode 陈旧问题（重启容器重新解析）；②挂载的 start-web.sh 需执行位

### 回归（苯并咪唑论文 balanced）
- 修复前：0 structures（10 候选全丢）；**修复后：8 保留（6 个正确 SMILES 的 standalone_compound + 2 个 markush 占位）+ 9 条留痕过滤**（reactant/intermediate/噪声各带原因）
- 视觉验证：红框精确圈住化合物 G（修复前错位到文字区）
- 前端：full pipeline toast 真实计数、Step 4 空态文案分流、Step 2 被过滤候选面板、UI 构建部署完成

### 已知遗留（后批）
- P1-2 页 scheme review 回填一致性；P1-3 predict_lock 卡死缓解；P1-7 孤儿子进程收养；markush 行（无 SMILES）的 UI 呈现策略；ID 抽取 "6a-r" 系列标签的语义处理；detection recall（page 2 化合物 B/C 漏检）

## 16. 四路工程化 review（2026-11-13 深夜）

独立四路（结构链路/前端部署/仓库卫生/AI 味审计）共产出 P0×7、P1×5、清单百余项。已实施：

**P0 修复**
- ledger 接线三处断裂（process_segment 未收 ledger、补页 submit 漏参、匹配键在补框重试后失配）——此前"18/18 终态"实为行合成假数据；修复后 19/19 真实终态
- proxy 并发竞态：os.environ 翻转在页线程池下永久泄漏/丢代理——改为 require_openai() 工厂内 http_client 显式绑定代理
- ledger.open_unit bbox int 归一（numpy bbox 会炸 json）
- 前端被过滤面板键名大小写（PAGE_NUM/STRUCTURE_FILTER_REASON）
- CSV 消毒与 dtype=str 读回的永久污染（读侧对称剥离 + 前缀收窄为 =/＠/tab）
- mergeStructureRecordLists 去重键含 sequence 致永不命中

**P1 修复**：6 处 pd.read_csv 统一 _read_csv_strict；open-job/URL 恢复补 filtered_records；空态文案两处统一；拒绝行泄漏主表分流；crop_check 矛盾和解

**工程化**：AI 味注释清理（App.tsx 15 处、main.py 20 行、pipeline.py 9 行、structure_parser 8 处、work_queue __import__、%-格式化）；死代码/死分支；仓库卫生（死测试脚本、残留 log、runtime_models 空壳、.gitignore 补 override）

**新发现资产风险（待用户决策）**：skills/name_resolution、frontend/ui/scripts/precompress-assets.js、11 个 utils、training 多个模块被活引用但从未 git add——fresh clone 会断链；另有 upload_to_hf*.sh 含明文 HF token 建议轮换。

**回归**：全仓 py_compile ✓、tsc+前端构建 ✓、端到端 11 条保留 + ledger 19/19 真实终态 ✓。备份：code_backup_20261113.tar.gz。
