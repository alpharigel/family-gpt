from dataclasses import dataclass
from .prompt import BASE_PROMPT
from ..agent_type import StreamlitAgentType

@dataclass
class ChatAgentConfig():
    prompt: str = BASE_PROMPT
    agent_type: StreamlitAgentType = StreamlitAgentType.CONVERSATION_CHAIN

