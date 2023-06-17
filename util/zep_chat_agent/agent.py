import streamlit as st
from util.message import message, message_style

import requests
import datetime
import pickle
from dotenv import load_dotenv
import streamlit_google_oauth as oauth
import os
from util.db_postgress import FamilyGPTDatabase, AgentConfig
from langchain.memory.chat_message_histories import ZepChatMessageHistory

load_dotenv()

client_id = os.environ["GOOGLE_CLIENT_ID"]
client_secret = os.environ["GOOGLE_CLIENT_SECRET"]
redirect_uri = os.environ["GOOGLE_REDIRECT_URI"]
db_connection_string = os.environ["DATABASE_URL"]
ZEP_API_URL = os.environ["ZEP_API_URL"]

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.schema import messages_to_dict, messages_from_dict
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from .prompt import BASE_PROMPT

from ..streamlit_agent import StreamlitAgent

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ZepChatAgent(StreamlitAgent):

    def __init__(self, session_id: str, user_prompt: str = BASE_PROMPT, ai_prefix: str = "AI"):        
        """Generate a ConversationChain with Zep chat history."""

        self.ai_prefix = ai_prefix
        self.user_prompt = user_prompt

        logger.info(f"Generating ConversationChain with Zep chat history for session {session_id}")

        # setup zep chat history
        self.zep_chat_history = ZepChatMessageHistory(session_id=session_id, url=ZEP_API_URL)

        self.memory = ConversationBufferMemory(
            memory_key="history", chat_memory=self.zep_chat_history, ai_prefix=self.ai_prefix
        )

        # initialize the agent
        self.llm = ChatOpenAI(temperature=0.8)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.user_prompt),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        ) 

        self.agent = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm)

    def run(self, *args, **kwargs):
        return self.agent.run(*args, **kwargs)
    
    def rebuild_memory():
        pass
