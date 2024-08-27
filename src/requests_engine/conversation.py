from typing import TypedDict, Any


class MessageContent(TypedDict):
    type: str
    text: str


class Message(TypedDict):
    role: str
    content: list[MessageContent]


class Conversation:
    def __init__(self):
        self.system_prompt: str = None
        self.messages: list[Message] = []

    @classmethod
    def with_initial_message(cls, system_prompt: str, role: str, content: str) -> "Conversation":
        instance = cls()
        instance.system_prompt = system_prompt
        instance.add_message(role, content)
        return instance

    def add_message(self, role: str, content: str):
        self.messages.append(
            {
                "role": role,
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        )

    def to_openai_format(self) -> list[Any]:
        messages = [{"role": "user", "content": self.system_prompt}]
        messages.extend(
            [{"role": message["role"], "content": message["content"][0]["text"]} for message in self.messages]
        )
        return messages

    def to_anthropic_format(self) -> list[Any]:
        return self.messages

    def __repr__(self):
        return str(self.messages)
