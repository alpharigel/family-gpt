"""An abstract class for agents that can chat using Streamlit."""

from abc import ABC, abstractmethod
from .message import message, message_style
import streamlit as st

class StreamlitAgentConfig:
    pass


class StreamlitAgent(ABC):
    """An abstract class for agents that can chat using Streamlit."""

    agent_name: str
    config_data: StreamlitAgentConfig
    generated: list
    past: list
    prev_input: str
    submitted_input: str

    def __init__(self):
        """Initialize the agent."""

    @abstractmethod
    def run(self, user_input:str) -> str:
        """Given user input, run the chain"""

    @abstractmethod
    def streamlit_render(self) -> None:
        """Render the agent content in streamlit, such as messagess"""

    def submit(self):
        self.submitted_input = st.session_state.input_widget
        st.session_state.input_widget = ""

    def render_message_interface(self):
        st.title(f"{self.agent_name} Personal Assistant")

        st.text_input("You: ", key="input_widget", on_change=self.submit)
        user_input = self.submitted_input

        # If the user has submitted input, ask the AI
        if user_input and user_input != self.prev_input:
            self.run(user_input)
            self.prev_input = user_input

        message_style()

        # Display the messages
        if self.generated:
            for i in range(len(self.generated) - 1, -1, -1):
                message(self.generated[i])
                message(self.past[i], is_user=True)
