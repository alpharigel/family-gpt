from enum import Enum


class StreamlitAgentType(str, Enum):
    CONVERSATION_CHAIN = "Conversation Chain"
    CHAIN_WITH_ZEP = "Conversation Chain with Zep"
    ZEP_TOOLS = "Zep Tools"
