import streamlit as st

from dotenv import load_dotenv
import streamlit_google_oauth as oauth
import os

load_dotenv()

client_id = os.environ["GOOGLE_CLIENT_ID"]
client_secret = os.environ["GOOGLE_CLIENT_SECRET"]
redirect_uri = os.environ["GOOGLE_REDIRECT_URI"]
db_connection_string = os.environ["DATABASE_URL"]
ZEP_API_URL = os.environ["ZEP_API_URL"]

from dataclasses import dataclass

from util.agent_manager import AgentManager

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from typing import Dict, Union


@dataclass
class MyState:
    superuser: bool
    overwrite_ask: bool = False
    prior_config_name: str = ""


def get_session_state() -> MyState:
    """Get the session state."""
    return st.session_state["state"]


from util.agent_type import StreamlitAgentType

AVAILABLE_AGENTS: Dict[str, StreamlitAgentType] = {
    "Simple Chat": StreamlitAgentType.CONVERSATION_CHAIN,
    "Chat with LT Memory": StreamlitAgentType.CHAIN_WITH_ZEP,
}


@st.cache_resource
def get_agent_manager(
    user_id: str, superuser: bool, agent_type: StreamlitAgentType
) -> AgentManager:
    """Get the agent manager for the given user_id and agent_type."""
    return AgentManager(user_id=user_id, superuser=superuser, agent_type=agent_type)


def main(user_id: str, superuser: bool = False):
    """
    Main function, runs the personal assistant
    """

    # Setup users environement

    # Initialize the session state
    if "state" not in st.session_state:
        state = MyState(superuser=superuser)
        st.session_state["state"] = state
        st.session_state.agent_managers = {}
    else:
        state = st.session_state["state"]  # type: MyState

    # Create a sidebar
    with st.sidebar:
        selected_agent = st.selectbox(
            "Select Agent", list(AVAILABLE_AGENTS.keys()), key="agent_name"
        )
    if selected_agent is None:
        selected_agent = list(AVAILABLE_AGENTS.keys())[0]

    agent_type = AVAILABLE_AGENTS[selected_agent]
    if selected_agent in st.session_state.agent_managers:
        manager = st.session_state.agent_managers[selected_agent]
    else:
        manager = get_agent_manager(
            user_id=user_id, superuser=superuser, agent_type=agent_type
        )
        st.session_state.agent_managers[selected_agent] = manager

    manager.streamlit_render()


if __name__ == "__main__":
    # Set the page title and favicon
    st.set_page_config(page_title=f"GPT Personal Assistant", page_icon=":tree:")

    query_params = st.experimental_get_query_params()
    extra = ""
    superuser = False
    if "superuser" in query_params:
        if "True" in query_params["superuser"]:
            extra = "?superuser=True"
            superuser = True

    # Create a login button using st.button
    login_info = oauth.login(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri + extra,
        app_name="Continue with Google",
        logout_button_text="Logout",
    )

    # Check if the user is logged in.
    if login_info:
        user_id, _user_email = login_info
        main(user_id, superuser=superuser)

    else:
        st.write("Please login")
