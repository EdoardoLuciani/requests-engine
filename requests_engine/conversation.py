class Conversation:
    def __init__(self):
        self.messages = []

    @classmethod
    def with_initial_message(cls, role, context):
        instance = cls()
        instance.add_message(role, context)
        return instance

    def add_message(self, role, content):
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