from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import dashscope
from qwen_llm import call_qwen

# ======================
# 定义 Agent State
# ======================

class AgentState(TypedDict):
    question: str
    docs: List[str]
    analysis: str
    critique: str
    final_answer: str


# ======================
# Sub-Agents
# ======================

def search_agent(state: AgentState):
    print("\n🔍 [Search Agent]")

    prompt = f"""
你是一个研究助理。
请针对以下问题，列出 3-5 条【事实性背景资料】，
不要给结论，只给事实要点：

问题：
{state['question']}
"""

    text = call_qwen(prompt)

    docs = [line for line in text.split("\n") if line.strip()]

    print("\n search document",docs)
    return {"docs": docs}


def analysis_agent(state: AgentState):
    print("\n🧠 [Analysis Agent]")

    prompt = f"""
你是一个分析型专家。
基于以下资料，给出你的【分析和判断】：

资料：
{state['docs']}

要求：
- 有条理
- 给出 2-3 个明确方向
"""

    analysis = call_qwen(prompt)

    print("\n analysis result:",analysis)
    return {"analysis": analysis}


def critic_agent(state: AgentState):
    print("\n🔨 [Critic Agent]")

    prompt = f"""
你是一个非常严格的批判者。
请针对下面的分析，提出问题和不足：

分析内容：
{state['analysis']}

要求：
- 指出漏洞
- 指出风险
- 指出不现实之处
"""

    critique = call_qwen(prompt)

    print("\n critic content:",critique)
    return {"critique": critique}


def final_agent(state: AgentState):
    print("\n✅ [Final Agent]")

    prompt = f"""
你是最终决策者。
请结合【分析】和【批判】，给出修正后的最终结论。

分析：
{state['analysis']}

批判：
{state['critique']}

输出要求：
- 给出清晰建议
- 偏向可落地
"""

    final_answer = call_qwen(prompt)
    print("\n critic content:", final_answer)
    return {"final_answer": final_answer}


# ======================
# 4. 构建 LangGraph
# ======================

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("search", search_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("final", final_agent)

    graph.set_entry_point("search")

    graph.add_edge("search", "analysis")
    graph.add_edge("analysis", "critic")
    graph.add_edge("critic", "final")
    graph.add_edge("final", END)

    return graph.compile()


# ======================
# 5. Main
# ======================

if __name__ == "__main__":
    app = build_graph()

    result = app.invoke({
        "question": "2026 年 Web3 + AI 有哪些可落地的创业方向？"
    })

    print("\n==============================")
    print("🎉 最终输出")
    print("==============================")
    print(result["final_answer"])
