from dataclasses import dataclass
from ..agent_type import StreamlitAgentType
from .prompt import BASE_PROMPT
from ..streamlit_agent import StreamlitAgentConfig


@dataclass
class ZepChatAgentConfig(StreamlitAgentConfig):
    """Configuration for the ZepChatAgent"""

    zep_iteration: int = 0
    prompt: str = BASE_PROMPT
    agent_type: StreamlitAgentType = StreamlitAgentType.CHAIN_WITH_ZEP  # deprecated
