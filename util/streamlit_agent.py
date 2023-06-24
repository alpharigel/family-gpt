"""An abstract class for agents that can chat using Streamlit."""

from abc import ABC, abstractmethod


class StreamlitAgentConfig:
    pass


class StreamlitAgent(ABC):
    """An abstract class for agents that can chat using Streamlit."""

    agent_name: str
    config_data: StreamlitAgentConfig

    def __init__(self):
        """Initialize the agent."""
        pass

    @abstractmethod
    def run(self) -> str:
        """Given user input, run the chain"""
        pass

    @abstractmethod
    def streamlit_render(self) -> None:
        """Render the agent content in streamlit, such as messagess"""
        pass
