"""An abstract class for agents that can chat using Streamlit."""

from abc import ABC, abstractmethod

class StreamlitAgent(ABC):
    """An abstract class for agents that can chat using Streamlit."""

    def __init__(self):
        """Initialize the agent."""
        pass

    @abstractmethod
    def run(self) -> str:
        """Given user input, run the chain"""
        pass

    @abstractmethod
    def messages_to_display(self) -> tuple(str, str):
        pass

    @abstractmethod
    @property
    def ai_prefix(self) -> str:
        pass
