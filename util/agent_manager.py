"""Encapsulates the selection and updating of an agent"""

import streamlit as st
from .db_postgress import FamilyGPTDatabaseAgents
import datetime

from .agent_config import AgentConfig
import copy

from .streamlit_agent import StreamlitAgent
from .agent_type import StreamlitAgentType
from .chat_agent.agent import ChatAgent
from .zep_chat_agent.agent import ZepChatAgent
from .zep_with_tools.agent import ZepToolsAgent


from typing import Union, Callable


@st.cache_resource
def generate_agent(
    user_id: str,
    agent_id: int,
    config_data: dict,
    agent_name: str,
    _update_agent_in_db: Callable[[StreamlitAgent], None],
    agent_type: StreamlitAgentType,
) -> StreamlitAgent:
    """Generate an agent based on the agent type."""

    if agent_type == StreamlitAgentType.CONVERSATION_CHAIN:
        return ChatAgent(
            user_id=user_id,
            agent_id=agent_id,
            config_data=config_data,
            agent_name=agent_name,
            update_agent_in_db=_update_agent_in_db,
        )
    elif agent_type == StreamlitAgentType.CHAIN_WITH_ZEP:
        return ZepChatAgent(
            user_id=user_id,
            agent_id=agent_id,
            config_data=config_data,
            agent_name=agent_name,
            update_agent_in_db=_update_agent_in_db,
        )
    elif agent_type == StreamlitAgentType.ZEP_TOOLS:
        return ZepToolsAgent(
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
    def __init__(self, user_id: str, superuser: bool, agent_type: StreamlitAgentType):
        """initialize the manager"""

        # load the available agents from the database

        # pull the users config from the database
        database = FamilyGPTDatabaseAgents()
        database.create_tables()

        configs = database.load_configs(
            user_id=user_id, superuser=superuser, agent_type=agent_type
        )  # type: dict[str, AgentConfig]
        self.configs = configs
        self.database = database
        self.user_id = user_id
        self.superuser = superuser
        self.agent_type = agent_type

        # if no configs for this user, initialize with default config
        if len(self.configs) == 0:
            config_name = "Base"
            agent_name = "AI"
            config_data = dict()
            agent_config = database.save_config(
                user_id,
                config_name,
                config_data.__dict__,
                superuser,
                agent_name,
                agent_type=agent_type,
            )
            self.configs[config_name] = agent_config

        # get the most recently updated config
        self.selected_config = sorted(
            self.configs.values(), key=lambda x: x.update_date, reverse=True
        )[0]

        # create the agent
        agent_id = self.selected_config.agent_id
        config_data = self.selected_config.config_data
        agent_name = self.selected_config.agent_name
        agent = generate_agent(
            self.user_id,
            agent_id,
            config_data,
            agent_name,
            self.update_agent_in_db,
            agent_type=self.agent_type,
        )
        self.agent = agent

    def select_config(self, config_name):
        """Select the config by name"""
        self.selected_config = self.configs[config_name]

    def update_agent_in_db(self, agent: StreamlitAgent) -> None:
        agent_config = self.database.save_config(
            user_id=self.user_id,
            agent_name=agent.agent_name,
            config_name=self.selected_config.config_name,
            config_data=agent.config_data.__dict__,
            superuser=self.superuser,
            agent_type=self.agent_type,
        )

        self.selected_config = agent_config
        self.configs[agent_config.config_name] = agent_config
        st.success(
            f"Config with name '{agent_config.config_name}' updated successfully!"
        )

    def new_config(self):
        """Create a new config"""

        ai_name = self.selected_config.agent_name
        new_config_name = ai_name + "_" + datetime.datetime.now().strftime("%Y%m%d")
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
        agent_config = self.database.save_config(
            user_id,
            new_config_name,
            config_data,
            superuser,
            agent_name,
            agent_type=self.agent_type,
        )
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
                "Select agent config:",
                config_names,
                index=config_names.index(self.selected_config.config_name),
            )

            # If "New Configuration" is selected, ask user to enter a new config name
            if selected_prompt_name == "New Configuration":
                new_config_name = self.new_config()
                new_config = self.configs[new_config_name]
                st.success(f"New config '{new_config_name}' created!")

                # make sure the agent is up
                config_data = new_config.config_data
                agent = generate_agent(
                    self.user_id,
                    new_config.agent_id,
                    config_data,
                    new_config.agent_name,
                    self.update_agent_in_db,
                    agent_type=self.agent_type,
                )
                self.agent = agent
                self.update_agent_in_db(agent)

            elif selected_prompt_name != self.selected_config.config_name:
                # switch to the new config and load the messages

                self.select_config(selected_prompt_name)
                agent_id = self.selected_config.agent_id
                config_data = self.selected_config.config_data
                agent_name = self.selected_config.agent_name
                agent = generate_agent(
                    self.user_id,
                    agent_id,
                    config_data,
                    agent_name,
                    self.update_agent_in_db,
                    self.agent_type,
                )
                self.agent = agent

            st.session_state.setdefault("rename_config", False)
            if st.button("Rename config") or st.session_state.rename_config:
                st.session_state.rename_config = True
                new_config_name = st.text_input(
                    "Enter new config name:", value=self.selected_config.config_name
                )
                if (
                    st.button("Apply", key="apply rename am")
                    and new_config_name != self.selected_config.config_name
                ):
                    self.configs[new_config_name] = self.selected_config
                    self.database.rename_config(
                        self.user_id,
                        self.selected_config.config_name,
                        new_config_name,
                        self.superuser,
                        agent_type=self.agent_type,
                    )
                    del self.configs[self.selected_config.config_name]
                    self.selected_config.config_name = new_config_name
                    st.success(
                        f"Config with name '{new_config_name}' updated successfully!"
                    )
                    st.session_state.rename_config = False

                if st.button("Cancel", key="cancel rename am"):
                    st.session_state.rename_config = False

            else:
                st.session_state.rename_config = False

        self.agent.streamlit_render()
