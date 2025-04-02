class HistoryManager:
    def __init__(self):
        self.history = []

    def add_message(self, role, content):
        """Append a new message to the conversation history."""
        self.history.append({"role": role, "content": content})

    def get_history(self):
        """Return the full conversation history."""
        return self.history

    def clear(self):
        """Clear the conversation history if needed."""
        self.history = []
