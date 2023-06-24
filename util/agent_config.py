import datetime
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """
    Data class that contains the minimum data in an agent configuration
    """

    agent_id: int
    agent_name: str
    config_name: str
    config_data: dict  # Union[ChatAgentConfig, ZepChatAgentConfig, ZepToolsAgentConfig]  # usually contains 'prompt'
    update_date: datetime.datetime
    hidden: bool
    agent_type: str
