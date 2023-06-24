"""A langchain conversational chain that implements the streamlit_agent abstract class."""

import os

import logging
from typing import Callable

from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.prompts.prompt import PromptTemplate
from langchain.tools import Tool
from langchain.utilities import BingSearchAPIWrapper
from langchain.agents import AgentExecutor

from ..streamlit_agent import StreamlitAgent
from ..zep_chat_agent.agent import ZepChatAgent
from .config import ZepToolsAgentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ZepToolsAgent(ZepChatAgent):
    def __init__(
        self,
        user_id: str,
        agent_id: int,
        update_agent_in_db: Callable[[StreamlitAgent], None],
        config_data: dict,
        agent_name: str = "AI",
    ):
        # call initialization of parent class
        super().__init__(
            user_id=user_id,
            agent_id=agent_id,
            update_agent_in_db=update_agent_in_db,
            config_data=config_data,
            agent_name=agent_name,
        )


    def init_config_data(self, config_data:dict) -> ZepToolsAgentConfig:
        return ZepToolsAgentConfig(**config_data)  


    def init_llm(self):
        # initialize the agent
        self.llm = ChatOpenAI(temperature=0.8, client=None, model="gpt-3.5-turbo-0613")


    def clean_user_prompt(self, prompt: str) -> str:
        """Clean the user prompt."""
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
        """Compute the prompt for the AI to use."""
        partial_variables: dict = {
            "chat_search": self.zep_search,
        }
        if "{ai_prefix}" in self.config_data.prompt:
            partial_variables["ai_prefix"] = self.agent_name

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

    def create_agent(self) -> AgentExecutor:
        """Create the agent."""
        search = BingSearchAPIWrapper(
            bing_subscription_key=os.environ["BING_SUBSCRIPTION_KEY"],
            bing_search_url=os.environ["BING_SEARCH_URL"],
        )

        tools = [
            Tool.from_function(
                func=search.run,
                name="Search",
                description="useful for when you need to answer questions about current events"
                # coroutine= ... <- you can specify an async method if desired as well
            ),
        ]

        # self.agent = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm, verbose=True)
        return initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=self.memory,
            verbose=True,
            prompt=self.prompt,
        )





