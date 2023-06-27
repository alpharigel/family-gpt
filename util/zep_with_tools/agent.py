"""A langchain conversational chain that implements the streamlit_agent abstract class."""

import os

import logging
from typing import Callable

#from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate, MessagesPlaceholder)
from langchain.prompts.prompt import PromptTemplate
from langchain.tools import Tool
from langchain.utilities import BingSearchAPIWrapper
from langchain.agents import AgentExecutor
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
#from langchain.agents.openai_functions_multi_agent.base import OpenAIMultiFunctionsAgent

# import SystemMessage
from langchain.prompts.base import BasePromptTemplate

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
        return ChatOpenAI(temperature=0.8, client=None, model="gpt-3.5-turbo-0613")

    def clean_user_prompt(self, prompt: str) -> str:
        """Clean the user prompt."""
        if "{chat_search}" not in prompt:
            prompt = (
                "\nPotentially useful previous messages:\n{chat_search}\n"
                + prompt
            )
        #if "{chat_history}" not in prompt:
        #    prompt += "\n{chat_history}\n"
        if "{chat_history}" in prompt:
            prompt += prompt.replace("{chat_history}", "")
        if "{input}" in prompt:
            prompt = prompt.replace("{input}", "")
        # if '{input}' not in self.config_data.prompt:
        #    self.config_data.prompt += "\nHuman: {input}\n" + self.agent_name + ": "
        return prompt

    def compute_prompt(self, sys_prompt: str) -> BasePromptTemplate:
        """Compute the prompt for the AI to use."""
        partial_variables: dict = {
            "chat_search": self.zep_search,
        }
        if "{ai_prefix}" in self.config_data.prompt:
            partial_variables["ai_prefix"] = self.agent_name
        if "{current_datetime}" in self.config_data.prompt:
            partial_variables["current_datetime"] = self.get_current_datetime
        if "{current_location}" in self.config_data.prompt:
            partial_variables["current_location"] = 'Kentfield, CA'
        if "{tools}" in self.config_data.prompt:
            partial_variables["tools"] = ""
    

        sys_prompt_template = PromptTemplate.from_template(
            sys_prompt, partial_variables=partial_variables
        )
        full_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(prompt=sys_prompt_template),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        return full_template

    def create_agent(self) -> AgentExecutor:
        """Create the agent."""
        search = BingSearchAPIWrapper(
            bing_subscription_key=os.environ["BING_SUBSCRIPTION_KEY"],
            bing_search_url=os.environ["BING_SEARCH_URL"],
        )

        from langchain.tools.python.tool import PythonAstREPLTool
        python = PythonAstREPLTool()

        tools = [
            Tool.from_function(
                func=search.run,
                name="Search",
                description="useful for when you need to answer questions about current events"
                # coroutine= ... <- you can specify an async method if desired as well
            ),
            python,
        ]        

        # create the openai functions agent
        func_agent = OpenAIFunctionsAgent(
            tools=tools,
            llm=self.llm,
            prompt=self.prompt,
        )

        agent_executor = AgentExecutor(
            tools=tools,
            agent=func_agent,
            memory=self.memory,
            verbose=True, 
            handle_parsing_errors=True
        )
        # Do this so we can see exactly what's going on under the hood
        import langchain
        langchain.debug = True

        # self.agent = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.llm, verbose=True)
        return agent_executor





