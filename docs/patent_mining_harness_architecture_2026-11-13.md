# 专利文献挖掘 Harness 架构设计（2026-11-13）

综合四路独立深度调研：①先进布局无关方案（Uni-Parser/MolParser/RxnCaption/MolMole/PatCID/SureChEMBL 等 20+ 系统）②agentic 文档挖掘架构（Anthropic/VMAO/AgenticOCR/Temporal/LangGraph 模式）③专利挖掘需求空间（SureChEMBL/ChEMBL/HARVEST/Doc2SAR/MarkushGrapher 基准）④本仓场景硬编码审计。全部结论标注来源，见各调研原文（会话归档）。

## 一、用户指令 → 设计原则

| 指令 | 设计响应 | 证据 |
|---|---|---|
| "极度泛化，不写具体示例" | prompt 只给通用视觉从属判据（零示例）；分类字段非穷尽枚举+允许 unknown；转录必须逐字禁补全 | RxnCaption：BROS 让 VLM 出坐标 F1=0.3，预画框零示例 81.0；Borchmann：被迫格式诱发拒答 |
| "靠 harness 能力" | 空间定位全确定性（检测器+膨胀区域+索引叠加回注），VLM 只做零示例转录+归属+分类 | Set-of-Mark/OmniParser/BIVP 范式；MolMole：砍外部布局解析反而更准 |
| "各种版式都要能处理" | 版式多样性回归矩阵（7 轴分层，最差格≥最优格×0.8 才发版） | RxnCaption-11k 四拓扑均衡；MolDet-Bench 分桶 |
| "上下文管理" | 文档上下文=结构化账本（实体行/引用行/残留行），每单元拿投影不拿全文 | Anthropic 多 agent 教训：retrieve-not-carry |
| "目标跟随" | 单元目标账本：每页/每图/每段一个目标，FSM 恰一终态（含 FILTERED(原因)），缺口驱动补采 | VMAO 的子问题 DAG+5 停止条件 |
| "纠错" | 免费验证池前置（RDKit/文本层互查）→策略阶梯→人工队列；每次尝试落审计 | 按字段风险定价（自一致性轮数差异化已是本仓正确先例） |
| "不漏" | 覆盖漏斗：页→检出→目标→终态计数；文档完成=漏斗闭合 | 账本+funnel+终态语义+缺口补采四件套 |
| "原子级 API 交互" | 幂等键写入 hash(doc,unit,skill,attempt,content_hash)+宿主 upsert；多行写事务/outbox | Temporal/幂等架构原则 |
| "skills、环境专业化" | 技能网关：能力声明+环境需求+健康检查+成本档案+路由熔断 | LiteLLM 网关模板；MCP Registry server.json |

## 二、目标架构

```
宿主(FastAPI+DB) ◀── 原子写网关 W（幂等键 upsert/outbox/审计回执）
挖掘 HARNESS（durable 编排层）
  D 编排器(Plan-Execute-Verify-Replan, checkpoint) ─ B 单元目标账本+文档上下文 ─ C 缺口补采
  E 纠错引擎(免费验证池→策略阶梯→HITL 仲裁 G)
  F 技能网关(声明环境/健康/成本；路由 MolNexTR↔本地 VLM↔云端)
  [视觉调用][OCSR][检测器][活性抽取] + [确定性验证池: RDKit/文本层互查]
```

**已确认的现有资产**（调研实读）：`services/document_ledger.py`（实体/GapReason/Evidence 行）、`services/figure_fsm.py`（单图 FSM，MAX_ATTEMPTS=4）、`skills/registry.py`（/v1/skills 原子调用）——升级而非重建。本轮新增 `utils/extraction_ledger.py`（单元账本 v1，已在结构链路端到端运行）。

## 三、输出契约升级（下一版视觉调用）

采用"先转录、后分类"的片段级契约（调研报告 4.2 全文可直接使用）：
每框输出 fragments[]{verbatim, placement, attached, kind(非穷尽)} + region_fragments[]{..., relates_to}。
下游：标注=attached 片段按阅读顺序拼接；幻觉校验=verbatim 必须命中 PDF 文本层（免模型）；`record_box/record_id` 退役（零消费者）。
跨页/分离标注：身份 join（InChI/OPSIN 名称/指称串精确匹配），仅歧义升级 VLM 并排仲裁——不做跨页视觉魔法。

## 四、分阶段路线（合并三份路线，每阶段独立验证）

| 阶段 | 内容 | 验证门 |
|---|---|---|
| 0 可观测化（本轮已起步） | 单元账本+覆盖事件接通结构链路 | 苯并咪唑回归：18/18 单元终态 ✓ |
| 1 去场景化文案 | 硬编码审计清单逐项中性化；record_box 退役；ID_EVIDENCE 前缀控制流改显式字段 | 20 张多版式集自一致率不降；literature_eval 不回退 |
| 2 转录契约替换 role 枚举 | scheme/ID 调用改 fragments 契约；role 由规则派生；准入改 ID 可用性+策略参数 | 五类版式矩阵首版全绿；行级差集人工抽检 |
| 3 覆盖漏斗统一 | document_ledger+extraction_ledger 合并为任务级账本；API 暴露 stats | 注入缺口测试一致率>95% |
| 4 原子写网关 | 幂等键 upsert；kill -9 重放零重复零丢失 | 重放测试脚本化 |
| 5 纠错引擎+预算 | 策略阶梯+免费验证池+审计行；预算/停止条件一等公民 | 修复率↑且 token≤1.5× 基线 |
| 6 持久化编排 | LangGraph+Postgres（或 DBOS）；Celery 降为执行器；单元级断点续跑 | 中途 kill 后续跑调用数≤1.05× |
| 7 专利输出空间 | 按差距清单：靶点/测定上下文→锚点→行级完备性(Table Recall)→Markush 通式本体→实施例→同族去重 | DocSAR-200 Table Recall + 实施例覆盖率入 CI |

## 五、本轮已落地（2026-11-13）

1. **utils/extraction_ledger.py**：单元账本 v1（open/transition/note_correction/finalize_from_rows/coverage），原子落盘 extraction_ledger.json；结构链路接线完成（18/18 终态验证）
2. **逐框转录机制**（compound_id/annotation_text verbatim）：泛化回归 10 条保留、9 ID、3 标注——零场景示例
3. **前端框顺序 XYXY 修复**（裁剪+叠框两处）；卡片侧条/高亮移除
4. **P0/P1 修复**（fragment 分支、_constants、CSV 注入+dtype、子进程兜底、proxy finally、静默页失败、imread/缺页留痕）+ 死代码清理 + README 修复 + 镜像重建（ultralytics 钉版）
5. 已知遗留：P1-2 页级 review 一致性、P1-3 锁卡死缓解、P1-7 孤儿子进程、阶段 1-7 路线、ID "6a-r" 系列标签语义、detection recall（B/C 漏检）
