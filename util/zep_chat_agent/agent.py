"""A langchain conversational chain that implements the streamlit_agent abstract class."""

import datetime
import logging
import os
import re
from typing import Callable, List

import streamlit as st
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ZepChatMessageHistory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AIMessage, HumanMessage
from zep_python import MemorySearchResult
from zep_python.exceptions import NotFoundError

from ..streamlit_agent import StreamlitAgent
from .config import ZepChatAgentConfig
from ..exception import FamilyGPTException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ZEP_API_URL = os.environ["ZEP_API_URL"]

def get_session_id(user_id: str, agent_id: int, iteration: int) -> str:
    env_name = os.environ["FAMILYGPT_ENV_NAME"]
    session_id = str(f"{env_name}_{user_id}_{agent_id}_{iteration}")  # 
    safe_string = re.sub(r'\W+', '_', session_id)
    return safe_string

class ZepChatAgent(StreamlitAgent):
    def __init__(
        self,
        user_id: str,
        agent_id: int,
        update_agent_in_db: Callable[[StreamlitAgent], None],
        config_data: dict,
        agent_name: str = "AI",
    ):
        self.user_id = user_id
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.config_data = self.init_config_data(config_data) # type: ZepChatAgentConfig
        self.update_agent_in_db = update_agent_in_db

        self.past = []
        self.generated = []

        # create the agent from component parts
        self.zep_chat_history = self.init_zep_chat_history(user_id, agent_id)
        self.memory = self.init_memory()
        self.llm = self.init_llm()
        self.config_data.prompt = self.clean_user_prompt(self.config_data.prompt)
        self.prompt = self.compute_prompt(self.config_data.prompt)
        self.agent = self.create_agent()
        self.init_messages()

    def init_zep_chat_history(self, user_id, agent_id) -> ZepChatMessageHistory:
        session_id = get_session_id(user_id, agent_id, self.config_data.zep_iteration)
        logger.info(
            "Generating ConversationChain with Zep chat history for session %s", 
            session_id
        )
        # Use a standard ConversationBufferMemory to encapsulate the Zep chat history
        return ZepChatMessageHistory(
            session_id=session_id, url=ZEP_API_URL
        )


    def init_memory(self) -> ConversationBufferMemory:
        return ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=self.zep_chat_history,
            ai_prefix=self.agent_name,
        )

    def init_config_data(self, config_data:dict) -> ZepChatAgentConfig:
        return ZepChatAgentConfig(**config_data)  

    def init_llm(self):
        # initialize the agent
        return ChatOpenAI(temperature=0.8, client=None)

    def clean_user_prompt(self, prompt: str) -> str:

        if "{chat_search}" not in prompt:
            prompt = (
                "\nPotentially useful previous messages:\n{chat_search}\n"
                + prompt
            )
        if "{chat_history}" not in prompt:
            prompt += "\n{chat_history}\n"
        # if '{input}' not in self.config_data.prompt:
        #    self.config_data.prompt += "\nHuman: {input}\n" + self.agent_name + ": "
        return prompt

    def compute_prompt(self, prompt):

        partial_variables: dict = {
            "chat_search": self.zep_search,
        }
        if "{tools}" in self.config_data.prompt:
            partial_variables["tools"] = ""
        if "{agent_scratchpad}" in self.config_data.prompt:
            partial_variables["agent_scratchpad"] = ""
        if "{ai_prefix}" in self.config_data.prompt:
            partial_variables["ai_prefix"] = self.agent_name
        if "{current_datetime}" in self.config_data.prompt:
            partial_variables["current_datetime"] = self.get_current_datetime()

        prompt_template = PromptTemplate.from_template(
            prompt, partial_variables=partial_variables
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(prompt=prompt_template),
                # MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        return prompt

    def create_agent(self) -> ConversationChain:
        return ConversationChain(
            memory=self.memory, prompt=self.prompt, llm=self.llm, verbose=True
        )
        

    def init_messages(self):
        for m in self.zep_chat_history.messages:
            if isinstance(m, HumanMessage):
                self.past.append(m.content)
            elif isinstance(m, AIMessage):
                self.generated.append(m.content)
            else:
                raise FamilyGPTException(f"Unknown message {m} with content {m.content}")

        if len(self.past) > 0:
            self.submitted_input = ""
            self.prev_input = self.past[-1]
        else:
            self.submitted_input = "Hello!"
            self.prev_input = ""
            self.submit()

    def zep_search(self) -> str:
        # given the chat_history, input add the additional variables we can format

        try:
            results = self.zep_chat_history.search(
                self.submitted_input
            )  # type: List[MemorySearchResult]
        except NotFoundError:
            return "No results found"

        parsed = ""
        for r in results:
            rm = r.message
            if rm is None:
                continue
            parsed += f"{rm['created_at']} - {rm['role']}: {rm['content']}\n"

        return parsed

    def apply_prompt(self, prompt: str):
        if self.config_data.prompt != prompt:
            self.config_data.prompt = self.clean_user_prompt(prompt)
            self.prompt = self.compute_prompt(self.config_data.prompt)
            self.agent = self.create_agent()

    def run(self, user_input):
        output = self.agent.run(user_input)

        self.past.append(user_input)
        # self.past_id.append(user_message_id)
        self.generated.append(output)
        # self.generated_id.append(ai_message_id)

    def get_current_datetime(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def streamlit_render(self):
        self.render_sidebar()
        self.render_message_interface()

    def render_sidebar(self):
        with st.sidebar:
            new_agent_name = st.text_input(
                "Agent Name", self.agent_name, key="agent_name_widget"
            )

            new_user_prompt = st.text_area(
                "Agent Prompt", self.config_data.prompt, key="prompt_widget", height=300
            )

            if (
                new_agent_name != self.agent_name
                or new_user_prompt != self.config_data.prompt
            ):
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
        session_id = get_session_id(self.user_id, self.agent_id, self.config_data.zep_iteration)

        self.zep_chat_history = ZepChatMessageHistory(
            session_id=session_id, url=ZEP_API_URL
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=self.zep_chat_history,
            ai_prefix=self.agent_name,
        )
        self.agent = ConversationChain(
            memory=self.memory, prompt=self.prompt, llm=self.llm, verbose=True
        )

        for user, ai in zip(self.past, self.generated):
            self.zep_chat_history.add_user_message(user)
            self.zep_chat_history.add_ai_message(ai)

        # self.zep_chat_history.zep_messages.pop()  # not sure this is goining to work. hopefully?
        # self.zep_chat_history.zep_messages.pop()

        self.prev_input = ""
        self.submitted_input = ""

        self.update_agent_in_db(self)


