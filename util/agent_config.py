import datetime
from dataclasses import dataclass
from typing import Union
from .chat_agent.config import ChatAgentConfig
from .zep_chat_agent.config import ZepChatAgentConfig
from .zep_with_tools.config import ZepToolsAgentConfig

@dataclass
class AgentConfig:
    """
    Data class that contains the minimum data in an agent configuration
    """

    agent_id: str
    agent_name: str
    config_name: str
    config_data: Union[ChatAgentConfig, ZepChatAgentConfig, ZepToolsAgentConfig]  # usually contains 'prompt'
    update_date: datetime.datetime
    hidden: bool