from dataclasses import dataclass
from ..agent_type import StreamlitAgentType
from .prompt import BASE_PROMPT

@dataclass
class ZepToolsAgentConfig:
    """Configuration for the ZepToolsAgent"""
    zep_iteration: int = 0
    prompt: str = BASE_PROMPT
    agent_type: StreamlitAgentType = StreamlitAgentType.ZEP_TOOLS
