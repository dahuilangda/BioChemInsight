# 结果工具栏与记录历史

参考官方交互说明：[SeaTable 的过滤、排序和分组](https://seatable.com/help/grouping-sorting-filter/)、[Airtable 分组](https://support.airtable.com/articles/7361540623-grouping-records-in-airtable)、[Airtable 记录修订历史](https://support.airtable.com/articles/3516802427-record-level-revision-history-in-airtable)。本次落实独立操作入口、视图规则、记录级历史和明确的恢复范围。

## 统一面板

Columns、Filter、Group、Sort、Search、View options、Add、More actions、Export 和 History 使用同一个 ReviewPopover。面板以触发按钮为锚点，并限制在结果区域的水平范围内；高度受可用窗口空间限制。相同按钮再次点击、点击外部、Esc 和关闭按钮使用一致的收起逻辑。键盘焦点离开非模态面板后关闭面板，Esc 返回触发按钮。

统一 32 px 工具栏控件、16 px 图标、5 px 控件圆角、面板边框和轻阴影。Add 只放新增记录和测量，Undo / Needs review / Find missing data 放到 More actions。行数、密度、行高和显示选项归到 View options。

## 视图规则

- 多条件过滤可选择匹配全部或任一条件，支持空值、文字匹配和数值比较；未填完的条件不隐藏数据。
- 多字段排序按优先级依次执行，可上下调整规则。数值点与“数值 ± 不确定度”按数值排序，类别和截断值保持文本，空值放在最后。
- 最多三层分组，显示各组总记录数，支持折叠、展开与优先级调整。字段相同但大小写或文本表示不同的组保持连续，不因次级排序交错。
- 手动拖动行在分组或排序启用时停用；这些操作只改变视图，不改写实验值或来源归属。
- 查询规则、行数和列布局在本浏览器按任务保存。它们不是共享协作视图，也不会产生数据编辑审计。
- 测量字段创建、改名检查结构和审计保留字段，避免覆盖 SMILES、记录 ID 等字段。

## History

从选中记录打开时，默认筛选该记录并展开最近一次变化；没有记录变化时可切换查看全部记录。记录改名后，范围选择器使用当前名称，旧事件保留当时的名称。

科学字段的前后值直接展示；来源和复核属性放在 Source and review details，Molfile 原文按需展开。导出遵循当前范围和筛选条件。

恢复单值、恢复记录与恢复版本分开。整版恢复放在 Version options 下，确认时明确覆盖范围。恢复期间暂时停止结果编辑，返回结果核对当前任务；旧任务的恢复响应不会覆盖新任务。进行中的手动画框解析跨越恢复操作时，不把旧结果写入恢复后的数据。

## 验证

17 组浏览器回归通过，覆盖桌面和手机的菜单位置、收起/焦点行为、AND/OR 过滤、排序优先级、多层分组、折叠和展开、重载保留、记录级历史、恢复和恢复期间编辑锁定，以及既有分页、PDF 导航、保存和导出。

纯规则测试覆盖 10,000 条记录、空值、截断结果、混合数值/文本、分组连续性、损坏的本地偏好和记录不被视图操作改写。浏览器使用隔离接口数据；另在公网 SR 任务仅执行视图和历史查看，没有保存记录或执行恢复。

证据目录：`output/unified_controls_qa_20261116/`。包含 browser-matrix.json、各测试日志、桌面/手机截图和规则检查结果。测试命令仍为 `python frontend/tests/run_workbench_qa.py`。
