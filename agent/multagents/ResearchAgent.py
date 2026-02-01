from Agent import Agent
from qwen_llm import call_qwen

class ResearchAgent(Agent):
    def run(self, message: str) -> str:
        print(f"\n🔍 {self.name} 正在研究...")

        prompt = f"""
以下是任务计划：
{message}

请补充关键背景知识、技术要点和必要细节。
        """.strip()

        return call_qwen(self.system_prompt, prompt)
