"""Encapsulates the selection and updating of an agent"""

import streamlit as st
from .db_postgress import FamilyGPTDatabase
import datetime

from .agent_config import AgentConfig
import copy

from .streamlit_agent import StreamlitAgent
from .chat_agent.agent import ChatAgent, ChatAgentConfig
from .zep_chat_agent.agent import ZepChatAgent, ZepChatAgentConfig
from .zep_with_tools.config import ZepToolsAgentConfig
from .zep_with_tools.agent import ZepToolsAgent
from .agent_type import StreamlitAgentType

from typing import Union

@st.cache_resource
def generate_agent(
    _database: FamilyGPTDatabase, 
    user_id: str, 
    agent_id: str,
    agent_type: str,
    config_data: Union[ChatAgentConfig, ZepChatAgentConfig, ZepToolsAgentConfig], 
    agent_name: str, 
    _update_agent_in_db: callable,
    ) -> StreamlitAgent:
    """Generate an agent based on the agent type."""

    if agent_type == StreamlitAgentType.CONVERSATION_CHAIN:
        return ChatAgent(
            database=_database, 
            user_id=user_id, 
            agent_id=agent_id, 
            config_data=config_data, 
            agent_name=agent_name,
            update_agent_in_db=_update_agent_in_db,
            )
    elif agent_type == StreamlitAgentType.CHAIN_WITH_ZEP:
        return ZepChatAgent(
            database=_database, 
            user_id=user_id, 
            agent_id=agent_id, 
            config_data=config_data, 
            agent_name=agent_name,
            update_agent_in_db=_update_agent_in_db,
        )
    elif agent_type == StreamlitAgentType.ZEP_TOOLS:
        return ZepToolsAgent(
            database=_database, 
            user_id=user_id, 
            agent_id=agent_id, 
            config_data=config_data, 
            agent_name=agent_name,
            update_agent_in_db=_update_agent_in_db,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    

"""Manages the selection and updating of an agent"""
class AgentManager:

    def __init__(self, database: FamilyGPTDatabase, user_id: str, superuser: bool):
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
            config_data = ChatAgentConfig()
            agent_config = database.save_config(user_id, config_name, config_data, superuser, agent_name)
            self.configs[config_name] = agent_config # type: dict[str, AgentConfig]

        # get the most recently updated config
        self.selected_config = sorted(self.configs.values(), key=lambda x: x.update_date, reverse=True)[0]
        
        # create the agent
        agent_id = self.selected_config.agent_id
        agent_type = self.selected_config.config_data.agent_type
        config_data = self.selected_config.config_data
        agent_name = self.selected_config.agent_name
        agent = generate_agent(self.database, self.user_id, agent_id, agent_type, config_data, agent_name, self.update_agent_in_db)
        self.agent = agent

        
    def select_config(self, config_name):
        """Select the config by name"""
        self.selected_config = self.configs[config_name]

    def update_agent_in_db(self, agent: StreamlitAgent):
        agent_config = self.database.save_config(
            user_id=self.user_id,
            agent_name=agent.agent_name,
            config_name=self.selected_config.config_name,
            config_data=agent.config_data,
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
        user_id = self.user_id
        superuser = self.superuser
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
        with st.sidebar:
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
                agent_type = new_config.config_data.agent_type
                config_data = new_config.config_data
                agent = generate_agent(self.database, self.user_id, new_config.agent_id, agent_type, config_data, new_config.agent_name, self.update_agent_in_db)
                self.agent = agent
                self.update_agent_in_db(agent)

            elif selected_prompt_name != self.selected_config.config_name:
                # switch to the new config and load the messages

                self.select_config(selected_prompt_name)
                agent_id = self.selected_config.agent_id
                agent_type = self.selected_config.config_data.agent_type
                config_data = self.selected_config.config_data
                agent_name = self.selected_config.agent_name
                agent = generate_agent(self.database, self.user_id, agent_id, agent_type, config_data, agent_name, self.update_agent_in_db)


            agent_type = self.selected_config.config_data.agent_type
            new_agent_type = st.selectbox(
                "Select agent type:", 
                [StreamlitAgentType.CONVERSATION_CHAIN, StreamlitAgentType.CHAIN_WITH_ZEP, StreamlitAgentType.ZEP_TOOLS], 
                key="agent_type_select_widget",
                index=0 if agent_type == StreamlitAgentType.CONVERSATION_CHAIN else 1
            )

            if new_agent_type != agent_type:
                self.selected_config.config_data.agent_type = new_agent_type
                agent_type = new_agent_type
                agent_id = self.selected_config.agent_id
                config_data = self.selected_config.config_data
                agent_name = self.selected_config.agent_name
                agent = generate_agent(self.database, self.user_id, agent_id, agent_type, config_data, agent_name, self.update_agent_in_db)
                self.agent = agent
                self.update_agent_in_db(agent)

            st.session_state.setdefault("rename_config", False)
            if st.button("Rename config") or st.session_state.rename_config:
                st.session_state.rename_config = True
                new_config_name = st.text_input("Enter new config name:", value=self.selected_config.config_name)
                if st.button("Apply", key="apply rename am") and new_config_name != self.selected_config.config_name:
                    self.configs[new_config_name] = self.selected_config
                    self.database.rename_config(self.user_id, self.selected_config.config_name, new_config_name, self.superuser)
                    del self.configs[self.selected_config.config_name]
                    self.selected_config.config_name = new_config_name
                    st.success(f"Config with name '{new_config_name}' updated successfully!")
                    st.session_state.rename_config = False

                if st.button("Cancel", key="cancel rename am"):
                    st.session_state.rename_config = False

            else:
                st.session_state.rename_config = False


        
