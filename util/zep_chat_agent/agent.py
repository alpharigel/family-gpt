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
from zep_python.exceptions import NotFoundError

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
    


"""A langchain conversational chain that implements the streamlit_agent abstract class."""

from ..streamlit_agent import StreamlitAgent
from ..message import message, message_style

import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

from .prompt import BASE_PROMPT
from ..db_postgress import FamilyGPTDatabase
from .config import ZepChatAgentConfig

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ZepChatAgent(StreamlitAgent):

    def __init__(
            self, 
            database: FamilyGPTDatabase, 
            user_id: str, 
            agent_id: str, 
            update_agent_in_db: callable,
            config_data: ZepChatAgentConfig = ZepChatAgentConfig(), 
            agent_name: str = "AI",
            ):
        
        self.database = database
        self.user_id = user_id
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.config_data = config_data
        self.update_agent_in_db = update_agent_in_db

        session_id = f"{self.agent_id}_{self.config_data.zep_iteration}"

        self.past = []
        self.past_id = []
        self.generated = []
        self.generated_id = []

        logger.info(f"Generating ConversationChain with Zep chat history for session {session_id}")

        # Use a standard ConversationBufferMemory to encapsulate the Zep chat history
        self.zep_chat_history = ZepChatMessageHistory(session_id=session_id, url=ZEP_API_URL)

        self.memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=self.zep_chat_history, ai_prefix=self.agent_name
        )

        # initialize the agent
        self.llm = ChatOpenAI(temperature=0.8)

        if '{chat_search}' not in self.config_data.prompt:
            self.config_data.prompt = "\nPotentially useful previous messages:\n{chat_search}\n" + self.config_data.prompt
        if '{chat_history}' not in self.config_data.prompt:
            self.config_data.prompt += "\n{chat_history}\n"
        if '{input}' not in self.config_data.prompt:
            self.config_data.prompt += "\nHuman: {input}\n" + self.agent_name + ": "

        partial_variables = {
            "chat_search": self.zep_search,
        }
        if '{tools}' in self.config_data.prompt:
            partial_variables["tools"] = ""
        if '{agent_scratchpad}' in self.config_data.prompt:
            partial_variables["agent_scratchpad"] = ""            
        if '{ai_prefix}' in self.config_data.prompt:
            partial_variables["ai_prefix"] = self.agent_name

        self.partial_variables = partial_variables
        self.prompt = ChatPromptTemplate.from_template(self.config_data.prompt, partial_variables=self.partial_variables)
        #self.prompt = ChatPromptTemplate.from_template(self.config_data.prompt)
        #self.prompt = ZepChatPromptTemplate(template=self.config_data.prompt, zep_memory=self.zep_chat_history,
        #                                        input_variables=["input", "chat_history"],
        #                                        messages=BaseMessage(self.config_data.prompt))

        self.agent = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm, verbose=True)

        for m in self.zep_chat_history.messages:
            if isinstance(m, HumanMessage):
                self.past.append(m.content)
            elif isinstance(m, AIMessage):
                self.generated.append(m.content)
            else:
                raise Exception(f"Unknown message {m} with content {m.content}")
            
        if len(self.past) > 0:
            self.submitted_input = ''
            self.prev_input = self.past[-1]
        else:
            self.submitted_input = 'Hello!'
            self.prev_input = ''

    def zep_search(self) -> str:
        # given the chat_history, input add the additional variables we can format

        try:
            results = self.zep_chat_history.search(self.submitted_input)
        except NotFoundError as e:
            return "No results found"
        
        parsed = ""
        for r in results:
            rm = r.message
            parsed += f"{rm['created_at']} - {rm['role']}: {r.message['content']}\n"

        return parsed

    def apply_prompt(self, prompt: str):
        if self.config_data.prompt != prompt:
            self.config_data.prompt = prompt
            #self.prompt = ChatPromptTemplate.from_template(prompt)
            self.prompt = ChatPromptTemplate.from_template(self.config_data.prompt, partial_variables=self.partial_variables)
            #self.prompt = ZepChatPromptTemplate(template=self.config_data.prompt, zep_memory=self.zep_chat_history,
            #                                    input_variables=["input", "chat_history"])
            self.agent = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm, verbose=True)

    def run(self, user_input):
        output = self.agent.run(user_input)

        self.past.append(user_input)
        # self.past_id.append(user_message_id)
        self.generated.append(output)
        # self.generated_id.append(ai_message_id)
    
    def streamlit_render(self):
        self.render_sidebar()
        self.render_message_interface()

    def render_sidebar(self):
        with st.sidebar:

            new_agent_name = st.text_input("Agent Name", self.agent_name, key="agent_name_widget")

            new_user_prompt = st.text_area(
                "Agent Prompt", self.config_data.prompt, key="prompt_widget", height=300
            )

            if new_agent_name != self.agent_name or new_user_prompt != self.config_data.prompt:
                self.config_data.prompt = new_user_prompt
                self.agent_name = new_agent_name
                self.update_agent_in_db(self)
                
            if st.button("Apply", key="Prompt Apply"):
                # apply new prompt
                self.apply_prompt(new_user_prompt)

            if st.button("Current summary"):
                st.write(self.zep_chat_history.zep_summary)

            if st.button("Undo last message", key="undo_widget"):
                self.undo_last_message()

    def undo_last_message(self):
        self.past.pop()
        self.generated.pop()

        # unfortunatly, looks like we need to delete the memory and recreate for this to work...
        # zep_chat_history = ZepChatMessageHistory(session_id=self.agent_id, url=ZEP_API_URL)
        # very unhappy with this, looks like once something is deleted it can't be recovered
        # zep_chat_history.clear()
        # could work around by generating a new session id for zep.
        self.config_data.zep_iteration += 1
        session_id = f"{self.agent_id}_{self.config_data.zep_iteration}"

        self.zep_chat_history = ZepChatMessageHistory(session_id=session_id, url=ZEP_API_URL)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=self.zep_chat_history, ai_prefix=self.agent_name
        )
        self.agent = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm, verbose=True)

        for user, ai in zip(self.past, self.generated):
            self.zep_chat_history.add_user_message(user)
            self.zep_chat_history.add_ai_message(ai)

        #self.zep_chat_history.zep_messages.pop()  # not sure this is goining to work. hopefully?
        #self.zep_chat_history.zep_messages.pop()

        self.prev_input = ""
        self.submitted_input = ""

        self.update_agent_in_db(self)
        

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
                message(self.generated[i], key=str(i))
                message(self.past[i], is_user=True, key=str(i) + "_user")
