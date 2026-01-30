import subprocess
from typing import Any, Dict, List
import dashscope
from dashscope import Generation
import json

dashscope.api_key = "your_api_key"

# =====================================================
# 1️⃣ Shell Tool（真正执行）
# =====================================================
def execute_shell(command: str) -> str:
    result = subprocess.run(
        ["powershell", "-Command", command],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return f"ERROR:\n{result.stderr.strip()}"

    return result.stdout.strip()


# =====================================================
# 2️⃣ Command Guard（安全核心，必须保留）
# =====================================================
class CommandGuard:
    ALLOWED_PREFIX = [
        "pwd",
        "dir",
        "cd",
        "echo",
        "type",
        "Get-ChildItem",
        "Get-Location",
        "Get-Content",
        "ipconfig",
        "whoami",
        "python"
    ]

    FORBIDDEN = [
        "rm",
        "del",
        "format",
        "shutdown",
        "Restart-Computer",
        "Remove-Item",
        "diskpart",
        "reg delete"
    ]

    @classmethod
    def validate(cls, command: str) -> bool:
        lower = command.lower()
        if any(bad in lower for bad in cls.FORBIDDEN):
            return False
        return any(command.strip().startswith(p) for p in cls.ALLOWED_PREFIX)


# =====================================================
# 3️⃣ Function Schema（给 LLM 的 Tool 定义）
# =====================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_shell",
            "description": "Execute a safe Windows PowerShell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "PowerShell command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    }
]


# =====================================================
# 4️⃣ Function Calling Shell Agent（真实 LLM）
# =====================================================
class FunctionCallingShellAgent:
    def __init__(self):

        self.system_prompt = """
你是一个 Windows Shell 智能体。

规则：
- 你不能直接执行任何命令
- 如需执行命令，必须调用工具 execute_shell
- 禁止删除文件、修改系统、关机、破坏性操作
- 工具执行完成后，你需要基于结果给出自然语言总结
"""

    def run(self, user_goal: str):
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_goal}
        ]

        # ========== 第一次调用 LLM ==========
        response = Generation.call(
            model="qwen-turbo",
            messages=messages,
            tools=TOOLS,
            temperature=0.1,
            max_tokens=100,
            result_format="message",

        )

        msg = response.output.choices[0].message

        # ========== 是否触发 tool ==========
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            args = tool_call['function']['arguments']
            arg = json.loads(args)
            command = arg["command"]

            print(f"\n🧠 LLM 请求执行命令：{command}")

            result = execute_shell(command)

            print(f"📤 Shell 执行结果：\n{result}")
            messages.append(msg)
            messages.append({
                "role":"tool",
                "tool_name": tool_call['function']['name'],
                "content": result
            })
            # ========== 把 Tool 结果喂回 LLM ==========
            response = Generation.call(
                model="qwen-turbo",
                messages=messages,
                tools=TOOLS,
                temperature=0.1,
                max_tokens=100,
                result_format="message",
            )

            print(f"\n✅ Final Answer:\n{response.output.choices[0].message}")
            return

        # ========== 不需要调用工具 ==========
        print(f"\n✅ Final Answer:\n{msg.content}")


# =====================================================
# 5️⃣ 程序入口
# =====================================================
if __name__ == "__main__":
    agent = FunctionCallingShellAgent()

    while True:
        user_input = input("\n👉 你想让 Shell Agent 做什么？(exit 退出)\n> ")
        if user_input.lower() in {"exit", "quit"}:
            break

        agent.run(user_input)
