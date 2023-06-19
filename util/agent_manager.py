"""Encapsulates the selection and updating of an agent"""

import streamlit as st
from .db_postgress import FamilyGPTDatabase
import datetime

from enum import Enum
class AgentType(str, Enum):
    CONVERSATION_CHAIN = "Conversation Chain"
    CHAIN_WITH_ZEP = "Conversation Chain with Zep"

from .agent_config import AgentConfig
import copy

from .streamlit_agent import StreamlitAgent
from .chat_agent.agent import ChatAgent
from .zep_chat_agent.agent import ZepChatAgent


@st.cache_resource
def generate_agent(
    database: FamilyGPTDatabase, 
    user_id: str, 
    agent_id: str, 
    agent_type: str,
    user_prompt: str, 
    agent_name: str, 
    ) -> StreamlitAgent:
    """Generate an agent based on the agent type."""

    if agent_type == AgentType.CONVERSATION_CHAIN:
        return ChatAgent(
            database=database, 
            user_id=user_id, 
            agent_id=agent_id, 
            user_prompt=user_prompt, 
            agent_name=agent_name,
            )
    elif agent_type == AgentType.CHAIN_WITH_ZEP:
        return ZepChatAgent(agent_id, user_prompt, agent_name)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    

"""Manages the selection and updating of an agent"""
class AgentManager:

    def __init__(self, database: FamilyGPTDatabase, user_id: str, superuser: bool, default_prompt: str):
        """initialize the manager"""

        # load the available agents from the database
        
        # pull the users config from the database
        configs = database.load_configs(user_id=user_id, superuser=superuser) # type: dict[str, AgentConfig]
        self.configs = configs
        self.database = database
        self.user_id = user_id
        self.superuser = superuser

        # if no configs for this user, initialize with default config
        if len(self.configs) == 0:
            config_name = "Base"
            agent_name = "AI"
            config_data = {
                "prompt": default_prompt,
            }
            agent_config = database.save_config(user_id, config_name, config_data, superuser, agent_name)
            self.configs[config_name] = agent_config # type: dict[str, AgentConfig]

        # get the most recently updated config
        self.selected_config = sorted(self.configs.values(), key=lambda x: x.update_date, reverse=True)[0]
        
        # create the agent
        agent_id = self.selected_config.agent_id
        agent_type = self.selected_config.config_data.get("agent_type", AgentType.CONVERSATION_CHAIN)
        prompt = self.selected_config.config_data["prompt"]
        agent_name = self.selected_config.agent_name
        agent = generate_agent(self.database, self.user_id, agent_id, agent_type, prompt, agent_name)
        self.agent = agent

        
    def select_config(self, config_name):
        """Select the config by name"""
        self.selected_config = self.configs[config_name]

    def update_config(self, config_name, config_data):
        """Update the config with the given data"""
        self.selected_config = self.configs[config_name]
        self.selected_config.config_data = config_data
        self.selected_config.update_date = datetime.datetime.now()
        self.database.save_config(
            user_id=self.user_id,
            agent_name=self.selected_config.agent_name,
            config_name=self.selected_config.config_name,
            config_data=self.selected_config.config_data,
            superuser=self.superuser,
        )

    def update_agent_in_db(self, agent: StreamlitAgent):
        agent_config = self.database.save_config(
            user_id=self.user_id,
            agent_name=agent.agent_name,
            config_name=self.selected_config.config_name,
            config_data=self.selected_config.config_data,
            superuser=self.superuser,
        )

        self.selected_config = agent_config
        self.configs[agent_config.config_name] = agent_config
        st.success(f"Config with name '{agent_config.config_name}' updated successfully!")


    def new_config(self):
        """Create a new config"""

        ai_name = self.selected_config.agent_name
        new_config_name = (
            ai_name + "_" + datetime.datetime.now().strftime("%Y%m%d")
        )
        if new_config_name in self.configs:
            i = 1
            new_config_name = new_config_name + f" {i}"
            while new_config_name in self.configs:
                i += 1
                new_config_name = new_config_name[:-2] + f" {i}"

        # create a new config
        user_id = self.selected_config.user_id
        superuser = self.selected_config.superuser
        agent_name = self.selected_config.agent_name
        config_data = copy.deepcopy(self.selected_config.config_data)
        agent_config = self.database.save_config(user_id, new_config_name, config_data, superuser, agent_name)
        self.configs[new_config_name] = agent_config
        self.selected_config = agent_config
        return new_config_name

    def streamlit_render(self):
        """Display the configuration controls for the agents"""

        # new, list, rename, delete, select

        # Create a dropdown to select a prompt
        config_names = list(self.configs.keys())
        config_names.append("New Configuration")
        selected_prompt_name = st.selectbox(
            "Select agent config:", config_names, index=config_names.index(self.selected_config.config_name)
        )


        # If "New Configuration" is selected, ask user to enter a new config name
        if selected_prompt_name == "New Configuration":
            new_config_name = self.new_config()
            new_config = self.configs[new_config_name]
            st.success(f"New config '{new_config_name}' created!")

            # make sure the agent is up
            agent_type = new_config.config_data.get("agent_type", AgentType.CONVERSATION_CHAIN)
            prompt = new_config.config_data['prompt']
            agent = generate_agent(self.database, self.user_id, new_config.agent_id, agent_type, prompt, new_config.agent_name)
            self.agent = agent

        elif selected_prompt_name != self.selected_config.config_name:
            # switch to the new config and load the messages

            self.select_config(selected_prompt_name)
            agent_id = self.selected_config.agent_id
            agent_type = self.selected_config.config_data.get("agent_type", AgentType.CONVERSATION_CHAIN)
            prompt = self.selected_config.config_data['prompt']
            agent_name = self.selected_config.agent_name
            agent = generate_agent(self.database, self.user_id, agent_id, agent_type, prompt, agent_name)


        agent_type = self.selected_config.config_data.get("agent_type", AgentType.CONVERSATION_CHAIN)
        new_agent_type = st.selectbox(
            "Select agent type:", 
            [AgentType.CONVERSATION_CHAIN, AgentType.CHAIN_WITH_ZEP], 
            key="agent_type_select_widget",
            index=0 if agent_type == AgentType.CONVERSATION_CHAIN else 1
        )

        if new_agent_type != agent_type:
            self.selected_config.config_data["agent_type"] = new_agent_type
            agent_type = new_agent_type
            agent = generate_agent(self.database, self.user_id, agent_id, agent_type, prompt, agent_name)
            self.agent = agent
            self.update_agent_in_db(agent)

        st.session_state.setdefault("rename_config", False)
        if st.button("Rename config") or st.session_state.rename_config:
            st.session_state.rename_config = True
            new_config_name = st.text_input("Enter new config name:", value=self.selected_config.config_name)
            if st.button("Apply") and new_config_name != self.selected_config.config_name:
                self.selected_config.config_name = new_config_name
                self.update_config(new_config_name, self.selected_config.config_data)
                st.success(f"Config with name '{new_config_name}' updated successfully!")
                st.session_state.rename_config = False

            if st.button("Cancel"):
                st.session_state.rename_config = False

        else:
            st.session_state.rename_config = False


        
