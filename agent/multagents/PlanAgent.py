from qwen_llm import call_qwen
from Agent import Agent


class PlannerAgent(Agent):
    def run(self, message: str) -> str:
        print(f"\n🧠 {self.name} 正在规划任务...")

        prompt = f"""
用户问题：
{message}

请将任务拆解成清晰的执行步骤。
        """.strip()

        return call_qwen(self.system_prompt, prompt)
