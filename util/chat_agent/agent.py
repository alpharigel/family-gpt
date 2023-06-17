"""A langchain conversational chain that implements the streamlit_agent abstract class."""

from ..streamlit_agent import StreamlitAgent

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

class ChatAgent(StreamlitAgent):

    def __init__(self, database: FamilyGPTDatabase, user_id: str, agent_id: str, user_prompt: str = BASE_PROMPT, ai_prefix: str = "AI"):
        self.database = database
        self.user_id = user_id
        self.agent_id = agent_id
        self.ai_prefix = ai_prefix
        self.user_prompt = user_prompt

        self.past = []
        self.past_id = []
        self.generated = []
        self.generated_id = []

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(user_prompt),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )

        self.llm = ChatOpenAI(temperature=0.8)
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm, max_token_limit=1400, return_messages=True, ai_prefix=ai_prefix
        )
        # memory = ConversationBufferMemory(return_messages=True)
        self.agent = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm)

    def apply_prompt(self, prompt: str):
        if self.user_prompt != prompt:
            self.user_prompt = prompt
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(self.user_prompt),
                    MessagesPlaceholder(variable_name="history"),
                    HumanMessagePromptTemplate.from_template("{input}"),
                ]
            )
            self.agent = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm)

    def run(self, user_input):
        output = self.agent.run(user_input)

        # Save the message to the database        
        (user_message_id, ai_message_id) = agent.save_message(user_id=user_id, agent_id=agent_id, user=user_input, ai=output)

        self.past.append(user_input)
        self.past_id.append(user_message_id)
        self.generated.append(output)
        self.generated_id.append(ai_message_id)

    def messages_to_display(self):
        return (self.generated, self.past)
    
    def steamlit_content(self):

        with st.sidebar:

            if st.button("Current summary"):
                memory = self.memory
                messages = memory.chat_memory.messages
                previous_summary = ""
                my_summary = memory.predict_new_summary(messages, previous_summary)
                st.write(my_summary)

            if st.button("Clear messages"):
                st.session_state.past = []
                st.session_state["past_id"] = []
                st.session_state["generated"] = []
                st.session_state["generated_id"] = []
                st.session_state["last_input"] = ""
                st.session_state["submitted_input"] = ""
                self.rebuild_memory()

            if st.button("Load messages"):
                self.load_messages(user_id=self.user_id, agent_id=self.agent_id)

            pop_last = st.button("Undo last message", key="undo_widget")
            if pop_last:
                st.session_state.submitted_input = ""
                st.session_state.past.pop()
                st.session_state.generated.pop()
                self.database.delete_message(self.user_id, self.agent_id, st.session_state["past_id"].pop())
                self.database.delete_message(self.user_id, self.agent_id, st.session_state["generated_id"].pop())
                self.rebuild_memory()
                st.session_state["last_input"] = ""

    def rebuild_memory(self):
        self.memory
        self.memory.clear()
        for user, ai in zip(st.session_state.past, st.session_state.generated):
            self.memory.save_context({"input": user}, {"output": ai})


    def load_messages(self, user_id: str, agent_id: str):
        messages = self.database.load_messages(user_id=user_id, agent_id=agent_id)
        st.session_state.past.clear()
        st.session_state.generated.clear()
        st.session_state.past_id.clear()
        st.session_state.generated_id.clear()

        for m in messages:
            if m[0] == "HUMAN":
                st.session_state.past.append(m[1])
                st.session_state.past_id.append(m[2])
            elif m[0] == "AI":
                st.session_state.generated.append(m[1])
                st.session_state.generated_id.append(m[2])
            elif m[0] == "SYSTEM":
                st.write(f"Previous system message: {m[1]}")
            else:
                raise (
                    Exception(f"Unknown message type {m[0]} with content {m[1]}")
                )

        if len(st.session_state.past) > 0:
            st.session_state["last_input"] = st.session_state.past[-1]
        
        st.session_state["submitted_input"] = ""
        self.rebuild_memory()

    def save_message(self, user_id: str, agent_id: str, message: str):
        self.database.save_message(user_id=user_id, agent_id=agent_id, message=message)