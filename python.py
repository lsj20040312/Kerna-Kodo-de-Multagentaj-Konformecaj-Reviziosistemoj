"""
基于 LangGraph 的多 Agent 合规审计报告生成系统 (Demo)
架构：Supervisor -> [文档解析Agent, 流水分析Agent, 长链推理Agent] -> 报告编译Agent
核心长链推理：大额交易对手 -> 历史违规记录 -> 关联方清单 -> 判定违规占用
"""

import json
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 若未安装 langgraph，请执行: pip install langgraph

# ======================== 1. 全局状态定义 ========================
class AuditState(TypedDict):
    task: str                           # 用户任务描述
    sub_tasks: List[str]                # Supervisor 拆解的子任务列表
    # 专业 Agent 输出
    regulation_clauses: List[Dict]      # 文档解析 Agent: 提取的监管条款
    transactions: List[Dict]            # 模拟交易流水
    anomalies: List[Dict]               # 流水分析 Agent: 检测到的异常
    deep_chain_findings: List[Dict]     # 长链推理 Agent: 穿透式分析结果
    # 报告编译
    draft_findings: List[Dict]          # 初步汇总发现
    cross_validation_issues: List[str]  # 交叉验证矛盾信息
    final_report: str                   # 最终报告 (Markdown)
    human_review_required: bool         # 是否需要人工复审

# ======================== 2. 模拟数据与工具函数 ========================
# 模拟监管条款库
REGULATIONS = [
    {"id": "R001", "text": "单笔交易超过500万需报备，并穿透核查最终受益人。"},
    {"id": "R002", "text": "禁止向关联方提供无商业背景的资金拆借。"},
    {"id": "R003", "text": "大额交易对手若在过去12个月内有违规记录，必须进行强化尽职调查。"}
]

# 模拟交易流水
TRANSACTIONS = [
    {"id": "T001", "amount": 800_0000, "counterparty": "企业A", "date": "2026-01-15"},
    {"id": "T002", "amount": 200_0000, "counterparty": "企业B", "date": "2026-02-20"},
    {"id": "T003", "amount": 1200_0000, "counterparty": "企业C", "date": "2026-03-10"},
    {"id": "T004", "amount": 450_0000, "counterparty": "企业D", "date": "2026-03-28"},
]

# 模拟关联方清单
RELATED_PARTIES = ["企业A", "企业X", "企业Y"]

# 模拟外部违规记录库
VIOLATION_HISTORY = {
    "企业A": ["2025-06 因内幕交易被处罚"],
    "企业C": ["2025-11 因洗钱嫌疑被调查"],
    "企业B": []
}

def query_violation_history(party: str) -> List[str]:
    """多跳查询第一步：获取对手历史违规记录"""
    return VIOLATION_HISTORY.get(party, [])

def query_related_parties() -> List[str]:
    """多跳查询第二步：获取内部关联方清单"""
    return RELATED_PARTIES

def is_related_party(party: str) -> bool:
    return party in RELATED_PARTIES


# ======================== 3. Agent 节点实现 ========================
# ---------- 3.1 任务规划 Agent (Supervisor) ----------
def supervisor_node(state: AuditState) -> AuditState:
    """拆解主任务为多个可并行的子任务"""
    task = state["task"]
    print(f"\n[Supervisor] 接收任务: {task}")
    # 通常这里会用 LLM 动态拆解，此处模拟固定拆解
    sub_tasks = [
        "制度对标: 提取最新监管条款",
        "流水异常检测: 筛查超标交易、可疑对手",
        "关联交易穿透: 长链推理核查关联方违规占用",
        "风险评估定级"
    ]
    state["sub_tasks"] = sub_tasks
    print(f"[Supervisor] 子任务拆解: {sub_tasks}")
    return state

# ---------- 3.2 文档解析 Agent ----------
def doc_parse_agent(state: AuditState) -> AuditState:
    """模拟从监管 PDF 中提取条款"""
    print("\n[文档解析Agent] 正在解析监管文件...")
    # 模拟提取
    clauses = REGULATIONS
    state["regulation_clauses"] = clauses
    print(f"[文档解析Agent] 提取到 {len(clauses)} 条条款")
    return state

# ---------- 3.3 流水分析 Agent ----------
def transaction_analysis_agent(state: AuditState) -> AuditState:
    """对模拟交易进行统计异常检测"""
    print("\n[流水分析Agent] 正在扫描交易流水...")
    anomalies = []
    for tx in TRANSACTIONS:
        if tx["amount"] > 5_000_000:
            anomalies.append({
                "tx_id": tx["id"],
                "reason": f"金额超500万阈值 (实际{tx['amount']/10000}万)",
                "counterparty": tx["counterparty"]
            })
    state["transactions"] = TRANSACTIONS
    state["anomalies"] = anomalies
    print(f"[流水分析Agent] 发现 {len(anomalies)} 笔异常交易")
    return state

# ---------- 3.4 长链推理 Agent (多步逻辑) ----------
def deep_chain_agent(state: AuditState) -> AuditState:
    """
    执行多跳长链推理：
    对于每一笔大额交易 -> 查对手违规记录 -> 查关联方清单 -> 判定是否构成关联方违规占用
    这是描述中的核心长链推理
    """
    print("\n[长链推理Agent] 启动穿透式关联交易审查...")
    findings = []
    anomalies = state.get("anomalies", [])
    # 仅分析异常交易以聚焦风险
    for anomaly in anomalies:
        party = anomaly["counterparty"]
        print(f"  → 审查交易 {anomaly['tx_id']} 对手方: {party}")
        # 第一步：查违规记录
        violations = query_violation_history(party)
        print(f"    历史违规记录: {violations if violations else '无'}")
        # 第二步：查是否关联方
        is_related = is_related_party(party)
        print(f"    是否为关联方: {is_related}")
        # 第三步：综合判定
        risk_level = "低"
        reasoning = ""
        if is_related and violations:
            risk_level = "高"
            reasoning = f"关联方且存在历史违规({'; '.join(violations)})，疑似无商业背景资金占用，触及监管条款 R002、R003"
        elif is_related:
            risk_level = "中"
            reasoning = "属于关联方交易，需进一步核查商业实质"
        elif violations:
            risk_level = "中"
            reasoning = f"非关联方但有违规记录，建议强化尽职调查"
        else:
            reasoning = "异常金额但无关联或违规记录"

        findings.append({
            "tx_id": anomaly["tx_id"],
            "counterparty": party,
            "risk_level": risk_level,
            "reasoning": reasoning,
            "violations": violations,
            "is_related": is_related
        })
    state["deep_chain_findings"] = findings
    print(f"[长链推理Agent] 完成穿透分析，产出 {len(findings)} 条深度结论")
    return state

# ---------- 3.5 报告编译 Agent (含交叉验证) ----------
def report_compilation_agent(state: AuditState) -> AuditState:
    """汇总所有 Agent 输出，交叉验证矛盾，生成最终报告"""
    print("\n[报告编译Agent] 开始汇总数据并交叉验证...")
    clauses = state.get("regulation_clauses", [])
    anomalies = state.get("anomalies", [])
    deep_findings = state.get("deep_chain_findings", [])
    transactions = state.get("transactions", [])

    draft_findings = []
    cross_issues = []

    # 1. 将异常流水与长链推理结果合并
    for af in anomalies:
        # 找到对应的深度分析
        deep = next((d for d in deep_findings if d["tx_id"] == af["tx_id"]), None)
        draft_findings.append({
            "tx_id": af["tx_id"],
            "amount": next(tx["amount"] for tx in transactions if tx["id"] == af["tx_id"]),
            "counterparty": af["counterparty"],
            "anomaly_reason": af["reason"],
            "deep_risk": deep["risk_level"] if deep else "未分析",
            "deep_reasoning": deep["reasoning"] if deep else "",
        })

    # 2. 交叉验证：检查流水分析 Agent 和长链推理 Agent 结论是否一致
    for df in draft_findings:
        # 模拟矛盾：若流水分析认为异常但长链推理风险为"低"，触发交叉检查
        if df["deep_risk"] == "低":
            cross_issues.append(
                f"冲突：交易 {df['tx_id']} 金额超标但长链推理未发现关联或违规风险，建议人工复核交易背景。"
            )
        # 实际上这里还会比对条款适用性等

    # 3. 生成 Markdown 报告
    report = f"""# 季度合规审计报告 (自动生成)

## 一、制度对标
已提取最新监管条款 {len(clauses)} 条：
"""
    for c in clauses:
        report += f"- **{c['id']}**: {c['text']}\n"

    report += "\n## 二、交易流水异常检测\n"
    report += f"共分析 {len(transactions)} 笔交易，发现异常 {len(anomalies)} 笔：\n"
    for a in anomalies:
        report += f"- {a['tx_id']}: {a['reason']} (对手: {a['counterparty']})\n"

    report += "\n## 三、关联交易穿透分析 (长链推理)\n"
    for f in deep_findings:
        report += f"### 交易 {f['tx_id']} (对手: {f['counterparty']})\n"
        report += f"- 风险等级: **{f['risk_level']}**\n"
        report += f"- 推理依据: {f['reasoning']}\n"
        report += f"- 违规记录: {f['violations']}\n\n"

    report += "\n## 四、交叉验证与冲突\n"
    if cross_issues:
        for issue in cross_issues:
            report += f"- ⚠️ {issue}\n"
    else:
        report += "- 未发现明显矛盾。\n"

    report += "\n## 五、复审建议\n"
    high_risk_items = [f for f in deep_findings if f["risk_level"] == "高"]
    if high_risk_items:
        report += f"**高风险项 {len(high_risk_items)} 条，需立即停牌核查！**\n"
        state["human_review_required"] = True
    else:
        report += "无高风险项，常规复审即可。\n"
        state["human_review_required"] = False

    state["draft_findings"] = draft_findings
    state["cross_validation_issues"] = cross_issues
    state["final_report"] = report
    print("[报告编译Agent] 报告生成完毕。")
    return state

# ======================== 4. 构建 LangGraph 图 ========================
def build_graph():
    builder = StateGraph(AuditState)
    
    # 添加节点
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("doc_parse", doc_parse_agent)
    builder.add_node("tx_analysis", transaction_analysis_agent)
    builder.add_node("deep_chain", deep_chain_agent)
    builder.add_node("report_compilation", report_compilation_agent)
    
    # 定义流程: supervisor 拆解后，并行启动三个专业 Agent (这里使用顺序模拟并行效果)
    builder.set_entry_point("supervisor")
    builder.add_edge("supervisor", "doc_parse")
    builder.add_edge("supervisor", "tx_analysis")
    # 长链推理需要等待流水分析的结果（获取异常列表），所以从 tx_analysis 之后触发
    builder.add_edge("tx_analysis", "deep_chain")
    # 报告编译需等待所有专业 Agent 完成 (doc_parse 和 deep_chain 都结束后)
    builder.add_edge("doc_parse", "report_compilation")
    builder.add_edge("deep_chain", "report_compilation")
    builder.add_edge("report_compilation", END)
    
    # 编译图 (带内存检查点)
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph

# ======================== 5. 运行示例 ========================
if __name__ == "__main__":
    print("="*60)
    print("启动多 Agent 合规审计报告系统")
    print("="*60)
    
    graph = build_graph()
    
    # 初始状态
    initial_state: AuditState = {
        "task": "生成2026年Q1资产管理业务合规报告",
        "sub_tasks": [],
        "regulation_clauses": [],
        "transactions": [],
        "anomalies": [],
        "deep_chain_findings": [],
        "draft_findings": [],
        "cross_validation_issues": [],
        "final_report": "",
        "human_review_required": False
    }
    
    # 配置线程用于状态持久化
    config = {"configurable": {"thread_id": "audit-2026Q1"}}
    
    # 执行图
    final_state = graph.invoke(initial_state, config)
    
    # 输出最终报告
    print("\n" + "="*60)
    print(final_state["final_report"])
    
    # 模拟Token消耗统计
    total_tokens = 7_000_000  # 示例值
    print(f"\n[系统] 本次任务消耗约 {total_tokens/1e6:.1f}M Token (模拟)")
    print(f"[系统] 是否需要人工复审: {'是' if final_state['human_review_required'] else '否'}")
