from Agent import Agent
from qwen_llm import call_qwen

class WriterAgent(Agent):
    def run(self, message: str) -> str:
        print(f"\n✍️ {self.name} 正在撰写最终答案...")

        prompt = f"""
请根据以下信息，输出一份结构清晰、易于理解的最终回答：

{message}
        """.strip()

        return call_qwen(self.system_prompt, prompt)
