# Cyclopeptide2D

基于完整残基位姿联合优化的环肽二维结构生成器，面向化学结构图绘制与 MolSight 合成训练数据准备。

**版本状态：0.1.0，研究预览版。** 当前实现已完成三个审核参考结构的几何验证，尚未证明对广泛环肽化学空间的覆盖率，也未完成合成数据对 MolSight 的训练收益验证。

## 背景与设计目标

大型环肽常包含非天然氨基酸、N-甲基化、嵌入主环的小环和多个立体中心。固定规则多边形后单独安排侧链，容易造成拥挤、交叉或错误的翻转。Cyclopeptide2D 将完整氨基酸残基作为几何单元，同时优化骨架及侧链所在单元的位置，而非固定骨架后修补局部坐标。

本工具生成二维化学示意图，不计算三维构象或构象能量。朝向环内的侧链是合法布局；中心留白不是约束，也不是优化目标。非对称性通过低频非对称初始化及空间拥挤共同产生，不保证每个随机种子获得不同构图。

## 算法

1. 根据分子图识别主环及酰胺接口，切分完整残基。脯氨酸等嵌入小环保留在完整残基中。通过临时虚拟连接端口获取局部接口几何。
2. 为每个残基建立独立二维坐标，并用平面刚体位姿 `(tx, ty, θ)` 表示其位置。主环闭合连接以键长等式同时求解，不使用最后一条键的强制拼接。
3. 使用解析梯度，先用 L-BFGS 进行惩罚松弛，再用 SLSQP 求解闭环等式及接口角度不等式。
4. 用完整残基反射搜索离散构图选择。反射作用于残基全部原子，固定局部 N—羰基 C 弦端点；立体化学标签保持不变，楔线依据最终坐标重新生成。
5. 使用合法连接处的正旋转进行局部构象微调；通过活动约束生成逐步加入原子—原子及原子—键接触，联合调整所有残基位姿。
6. 检查全部键长、接口角度、非键接触、键交叉、小环侵入和 MOL 序列化的立体构型往返一致性。

硬检查与优化评分相互独立。优化器报告成功不等于布局合格，调用方应始终检查 `result.accepted` 或报告中的 `qa_pass`。

## 安装

要求 Python 3.10 或更高版本，以及 RDKit、NumPy、SciPy。建议使用独立虚拟环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

开发时可使用 `python -m pip install -e .`。研究测试及评估结果只保留本地，不随 Git 仓库发布。

本目录可以复制到其他位置独立安装；运行时不依赖 BioChemInsight 的数据目录、旧绘图模块、服务、模型权重或网络。参考 SMILES 位于 `examples/reviewed_references.json`。首次安装依赖可能需要网络。

## 命令行使用

```bash
cyclopeptide2d \
  --smiles 'N1CC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC(=O)NCC1=O' \
  --seed 0 --maxiter 350 --out ./result
```

亦可运行 `python -m cyclopeptide2d`。`--smiles` 接受单条明确的 SMILES；命令退出码 `0` 表示几何验收通过，`2` 表示输入不支持或布局验收失败。紧小的环可能无法同时满足所有角度与接触约束；生产使用应保留失败报告。

成功输出：

| 文件 | 用途 |
| --- | --- |
| `structure.svg` | 黑白矢量结构图，使用求解坐标及最终楔线 |
| `structure.mol` | 带二维坐标的分子结构 |
| `report.json` | 验收结果、随机种子、约束余量、指标和求解状态 |

失败时输出 `report.json`，不生成新的正式结构图。每次调用建议使用独立输出目录，避免混淆之前调用的文件。

## Python API

```python
import json
from pathlib import Path
from rdkit import Chem
from cyclopeptide2d import depict, render_svg

references = json.loads(Path('examples/reviewed_references.json').read_text())
result = depict(references['53']['smiles'], seed=0, maxiter=350)

if result.accepted:
    Path('seq53.svg').write_text(render_svg(result), encoding='utf-8')
    Chem.MolToMolFile(result.molecule, 'seq53.mol')
else:
    print(result.report)
```

`depict` 返回 `LayoutResult(molecule, report)`。解析失败、自由基、通配原子或断开的分子组分会抛出 `ValueError`；不支持的环拓扑及未通过几何验收的候选通过报告返回。未指定的立体化学不会被自动补全。内部布局键长为 30 坐标单位；SVG 导出尺寸可以调整。

`render_svg` 只接受已通过几何检查的结果，保留已有坐标。它提供基础黑白绘制；扫描噪声、裁切、字体变体等训练增强属于后续数据流水线。

## 支持范围与边界

- 支持可切分为完整残基链的环肽，包括目前已验证的头尾酰胺环和一个 ClAc—Cys 硫醚闭环实例。后者不代表支持所有硫醚或多环拓扑。
- 融合、桥连或残基间额外交联可能无法表示为单环刚体链；不支持时明确返回失败，不自动换用其他结构或绘图器。
- 局部小环采用 RDKit 原生描绘，不通过缩短键或自由拖动原子消除拥挤。
- 随机种子用于初始化；浮点优化及原生描绘结果可能随依赖版本变化。可复现实验需记录依赖版本、输入和种子。
- 几何通过不证明原始 SMILES 与专利身份一致。输入 gold 必须单独核对原图、序列、连接及绝对构型，不能只匹配分子量。
- MOL 立体往返检查不等于像素图的楔线、标签全部可辨认。批量发布前仍需最终图像可读性检查和抽样目视检查。

## 验证与合成数据

当前参考为 US20240287138A1 的 53、68、83，经原图及残基定义审核。撤销不合理的中心留白约束后，三个结构均通过现有几何检查。首批 14–28 残基的八条随机合成输入仅一条通过，其余因约束未满足或超时退出，扩展覆盖仍在改进。此集合用于诊断，不应当作广泛覆盖率证明。

测试覆盖残基原子分区、完整残基反射等距性、闭环与接口解析梯度、接触梯度、合法局部正旋转、平移不变性，以及中心占用不影响优化的回归。

合成训练数据建议分阶段执行：先审核输入化学结构，生成候选并保存失败原因，再执行图像可读性检查，最后建立独立验证集。发布清单应记录输入规范化 SMILES、随机种子、软件版本、图像摘要和验收报告。MolSight 评测应分别报告连接结构完全正确率和包含立体化学的完全正确率；指纹相似度不能代替二者。

## 目录

```text
cyclopeptide2d/
├── pyproject.toml
├── README.md
├── examples/reviewed_references.json
├── src/cyclopeptide2d/
│   ├── api.py          # 公共接口、输入准入、SVG 绘制
│   ├── cli.py          # 命令行
│   ├── chemistry.py    # 分子图分解与立体校验
│   ├── geometry.py     # 几何测量、交叉与小环检查
│   └── solver.py       # 联合连续/离散优化与微调
```
