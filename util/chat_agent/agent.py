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
from ..db_postgress import FamilyGPTDatabaseMessages
from .config import ChatAgentConfig
from typing import Callable

import os


class ChatAgent(StreamlitAgent):
    def __init__(
        self,
        user_id: str,
        agent_id: int,
        update_agent_in_db: Callable[[StreamlitAgent], None],
        config_data: dict,
        agent_name: str = "AI",
    ):
        self.database = FamilyGPTDatabaseMessages(os.environ["DATABASE_URL"])
        self.user_id = user_id
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.config_data = ChatAgentConfig(**config_data)  # type: ChatAgentConfig
        self.update_agent_in_db = update_agent_in_db

        self.past = []
        self.past_id = []
        self.generated = []
        self.generated_id = []

        if "{chat_history}" in self.config_data.prompt:
            self.config_data.prompt = self.config_data.prompt.replace(
                "{chat_history}", ""
            )

        if "{chat_search}" in self.config_data.prompt:
            self.config_data.prompt = self.config_data.prompt.replace(
                "{chat_search}", ""
            )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.config_data.prompt),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )

        self.llm = ChatOpenAI(temperature=0.8, model="gpt-3.5-turbo", client=None)
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1400,
            return_messages=True,
            ai_prefix=agent_name,
        )
        # memory = ConversationBufferMemory(return_messages=True)
        self.agent = ConversationChain(
            memory=self.memory, prompt=self.prompt, llm=self.llm
        )

        # make sure the database has the message table
        self.database.create_tables()

        # pull the users messages from the database
        self.load_messages()

    def apply_prompt(self, prompt: str):
        if self.config_data.prompt != prompt:
            self.config_data.prompt = prompt
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(self.config_data.prompt),
                    MessagesPlaceholder(variable_name="history"),
                    HumanMessagePromptTemplate.from_template("{input}"),
                ]
            )
            self.agent = ConversationChain(
                memory=self.memory, prompt=self.prompt, llm=self.llm
            )

    def run(self, user_input):
        output = self.agent.run(user_input)

        # Save the message to the database
        (user_message_id, ai_message_id) = self.database.save_message(
            user_id=self.user_id,
            agent_id=self.agent_id,
            user=user_input,
            ai=output,
        )

        self.past.append(user_input)
        self.past_id.append(user_message_id)
        self.generated.append(output)
        self.generated_id.append(ai_message_id)

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

            if st.button("Apply"):
                # apply new prompt
                self.apply_prompt(new_user_prompt)

                memory = self.memory
                memory.clear()
                for user, ai in zip(self.past, self.generated):
                    memory.save_context({"input": user}, {"output": ai})

            if st.button("Current summary"):
                memory = self.memory
                messages = memory.chat_memory.messages
                previous_summary = ""
                my_summary = memory.predict_new_summary(messages, previous_summary)
                st.write(my_summary)

            if st.button("Clear messages"):
                self.past = []
                self.past_id = []
                self.generated = []
                self.generated_id = []
                self.prev_input = ""
                self.submitted_input = ""
                self.rebuild_memory()

            if st.button("Load messages"):
                self.load_messages()

            if st.button("Undo last message", key="undo_widget"):
                self.past.pop()
                self.generated.pop()
                self.database.delete_message(
                    self.user_id, self.agent_id, self.past_id.pop()
                )
                self.database.delete_message(
                    self.user_id, self.agent_id, self.generated_id.pop()
                )
                self.rebuild_memory()
                self.prev_input = ""
                self.submitted_input = ""

    def rebuild_memory(self):
        self.memory
        self.memory.clear()
        for user, ai in zip(self.past, self.generated):
            self.memory.save_context({"input": user}, {"output": ai})

    def load_messages(self):
        messages = self.database.load_messages(
            user_id=self.user_id, agent_id=self.agent_id
        )
        self.past = []
        self.past_id = []
        self.generated = []
        self.generated_id = []

        for m in messages:
            if m[0] == "HUMAN":
                self.past.append(m[1])
                self.past_id.append(m[2])
            elif m[0] == "AI":
                self.generated.append(m[1])
                self.generated_id.append(m[2])
            elif m[0] == "SYSTEM":
                st.write(f"Previous system message: {m[1]}")
            else:
                raise (Exception(f"Unknown message type {m[0]} with content {m[1]}"))

        if len(self.past) > 0:
            self.submitted_input = ""
            self.prev_input = self.past[-1]
        else:
            self.submitted_input = "Hello!"
            self.prev_input = ""

        self.rebuild_memory()

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
