import datetime
from dataclasses import dataclass

@dataclass
class AgentConfig:
    """
    Data class that contains the minimum data in an agent configuration
    """

    agent_id: str
    agent_name: str
    config_name: str
    config_data: dict  # usually contains 'prompt'
    update_date: datetime.datetime
    hidden: bool