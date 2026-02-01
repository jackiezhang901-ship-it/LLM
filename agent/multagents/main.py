from PlanAgent import PlannerAgent
from ResearchAgent import ResearchAgent
from WriterAgent import WriterAgent
from OrchestrateAgents import MultiAgentSystem

if __name__ == "__main__":
    planner = PlannerAgent(
        name="Planner",
        system_prompt="你是一个擅长将复杂问题拆解成步骤的专家。"
    )

    researcher = ResearchAgent(
        name="Researcher",
        system_prompt="你是一个技术研究员，擅长补充背景知识和技术细节。"
    )

    writer = WriterAgent(
        name="Writer",
        system_prompt="你是一个技术写作者，擅长将复杂内容讲清楚。"
    )

    system = MultiAgentSystem(
        agents=[planner, researcher, writer]
    )

    result = system.run(
        "请解释 Transformer Decoder 的训练原理"
    )

    print("\n🎉 最终输出：\n")
    print(result)
