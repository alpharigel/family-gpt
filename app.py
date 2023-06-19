import streamlit as st

from dotenv import load_dotenv
import streamlit_google_oauth as oauth
import os
from util.db_postgress import FamilyGPTDatabase

load_dotenv()

client_id = os.environ["GOOGLE_CLIENT_ID"]
client_secret = os.environ["GOOGLE_CLIENT_SECRET"]
redirect_uri = os.environ["GOOGLE_REDIRECT_URI"]
db_connection_string = os.environ["DATABASE_URL"]
ZEP_API_URL = os.environ["ZEP_API_URL"]

from dataclasses import dataclass

from util.streamlit_agent import StreamlitAgent
from util.chat_agent.prompt import BASE_PROMPT
from util.agent_manager import AgentManager

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class MyState:
    agent_manager: AgentManager
    superuser: bool
    overwrite_ask:bool = False
    prior_config_name:str = ""
    agent: StreamlitAgent = None


def get_session_state() -> MyState:
    """Get the session state."""
    return st.session_state["state"]


def main(user_id: str, superuser: bool = False):
    """
    Main function, runs the personal assistant
    """

    # Setup users environement
    database = FamilyGPTDatabase(db_connection_string)
    database.create_tables()

    # Initialize the session state
    if "state" not in st.session_state:
        agent_manager = AgentManager(database, user_id, superuser, default_prompt=BASE_PROMPT)
        state = MyState(agent_manager=agent_manager, superuser=superuser)
        st.session_state["state"] = state
    else:
        state = st.session_state["state"] # type: MyState


    # Create a sidebar
    agent_manager = state.agent_manager
    agent_manager.streamlit_render()
    agent_manager.agent.streamlit_render()
    



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
