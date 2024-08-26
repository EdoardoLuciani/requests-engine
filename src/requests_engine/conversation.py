from typing import TypedDict


class MessageContent(TypedDict):
    type: str
    text: str


class Message(TypedDict):
    role: str
    content: list[MessageContent]


class Conversation:
    def __init__(self):
        self.messages: list[Message] = []

    @classmethod
    def with_initial_message(cls, role: str, content: str) -> "Conversation":
        instance = cls()
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

    def __repr__(self):
        return str(self.messages)
