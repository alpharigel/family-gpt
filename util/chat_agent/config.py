from dataclasses import dataclass
from .prompt import BASE_PROMPT
from ..agent_type import StreamlitAgentType
from ..streamlit_agent import StreamlitAgentConfig

@dataclass
class ChatAgentConfig(StreamlitAgentConfig):
    prompt: str = BASE_PROMPT
    agent_type: str = '' # deprecated
    zep_iteration: int = 0 # deprecated

