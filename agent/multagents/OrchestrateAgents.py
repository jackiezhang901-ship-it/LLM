class MultiAgentSystem:
    def __init__(self, agents):
        self.agents = agents

    def run(self, user_input: str) -> str:
        message = user_input

        for agent in self.agents:
            message = agent.run(message)

        return message
