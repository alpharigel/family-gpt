import streamlit as st
from util.message import message, message_style

import requests
import datetime
import pickle
from dotenv import load_dotenv
import streamlit_google_oauth as oauth
import os
from util.db_postgress import FamilyGPTDatabase, AgentConfig
from langchain.memory.chat_message_histories import ZepChatMessageHistory

load_dotenv()

client_id = os.environ["GOOGLE_CLIENT_ID"]
client_secret = os.environ["GOOGLE_CLIENT_SECRET"]
redirect_uri = os.environ["GOOGLE_REDIRECT_URI"]
db_connection_string = os.environ["DATABASE_URL"]
ZEP_API_URL = os.environ["ZEP_API_URL"]

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.schema import messages_to_dict, messages_from_dict
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from util.streamlit_agent import StreamlitAgent

from util.chat_agent.prompt import BASE_PROMPT

from util.chat_agent.agent import ChatAgent
from util.zep_chat_agent.agent import ZepChatAgent

from enum import Enum
class AgentType(str, Enum):
    CONVERSATION_CHAIN = "Conversation Chain"
    CHAIN_WITH_ZEP = "Conversation Chain with Zep"


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@st.cache_resource
def generate_agent(database: FamilyGPTDatabase, user_id: str, agent_id: str, agent_tpe: str, user_prompt: str, ai_prefix: str, session_id: str) -> StreamlitAgent:
    """Generate an agent based on the agent type."""

    if agent_tpe == AgentType.CONVERSATION_CHAIN:
        return ChatAgent(database, user_id, agent_id, user_prompt, ai_prefix)
    elif agent_tpe == AgentType.CHAIN_WITH_ZEP:
        return ZepChatAgent(session_id, user_prompt, ai_prefix)
    else:
        raise ValueError(f"Unknown agent type: {agent_tpe}")
    

def init_session_state(configs: dict, superuser: bool = False):
    """
    Initialize the session state variables
    """

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    if "generated_id" not in st.session_state:
        st.session_state["generated_id"] = []

    if "past_id" not in st.session_state:
        st.session_state["past_id"] = []

    if "submitted_input" not in st.session_state:
        st.session_state["submitted_input"] = "Hello!"

    if "last_input" not in st.session_state:
        st.session_state["last_input"] = ""

    if "overwrite_ask" not in st.session_state:
        st.session_state["overwrite_ask"] = False

    if "configs" not in st.session_state:
        st.session_state["configs"] = configs

    if "superuser" not in st.session_state:
        st.session_state["superuser"] = superuser

    if "prior_config_name" not in st.session_state:
        st.session_state["prior_config_name"] = ""

    if "loaded_from_db" not in st.session_state:
        st.session_state["loaded_from_db"] = False




import copy


def render_sidebar(configs: dict, user_id: str, superuser: bool, database: FamilyGPTDatabase) -> StreamlitAgent:
    """
    Render the sidebar with the prompt selection and the prompt saving

    Parameters
    ----------
    configs : dict
        The dictionary of agents, where the key is the name of the configuration and the value is the prompt itself
        also contains values 'agent_id' and 'update_date', 'agent_name'
    superuser : bool
        Whether the user is a superuser or not
    """

    # Create a dropdown to select a prompt
    config_names = list(configs.keys())
    config_names.append("New Configuration")
    selected_prompt_name = st.selectbox(
        "Select agent config:", config_names, key="agent_select_widget"
    )

    # If "New Configuration" is selected, ask user to enter a new config name
    if selected_prompt_name == "New Configuration":
        if st.session_state["prior_config_name"] not in configs:
            prior_config = AgentConfig(
                agent_id="",
                agent_name="AI",
                config_name="New_" + datetime.datetime.now().strftime("%Y%m%d"),
                config_data = {
                    "prompt": BASE_PROMPT,
                    "agent_type": AgentType.CONVERSATION_CHAIN,
                },
                update_date=datetime.datetime.now(),
                hidden=superuser,
            )
        else:
            prior_config = configs[st.session_state["prior_config_name"]]
        ai_name = prior_config.agent_name
        new_config_name = (
            ai_name + "_" + datetime.datetime.now().strftime("%Y%m%d")
        )
        if new_config_name in config_names:
            i = 1
            new_config_name = new_config_name + f" {i}"
            while new_config_name in config_names:
                i += 1
                new_config_name = new_config_name[:-2] + f" {i}"

        config_names.insert(-1, new_config_name)
        config_data = copy.deepcopy(prior_config.config_data)
        new_config = database.save_config(
            user_id=user_id, config_name=new_config_name, config_data=config_data, superuser=superuser, agent_name=ai_name
        )
        st.success(f"New config '{new_config_name}' created!")
        selected_prompt_name = new_config_name
        configs[new_config_name] = new_config
        
        # clear out the message history for new config
        st.session_state["past"] = []
        st.session_state["past_id"] = []
        st.session_state["generated"] = []
        st.session_state["generated_id"] = []
        st.session_state["submitted_input"] = ""
        st.session_state["last_input"] = ""
    elif selected_prompt_name != st.session_state["prior_config_name"]:
        # switch to the new config and load the messages
        config = configs[selected_prompt_name]
        agent_id = config.agent_id
        if "agent" not in st.session_state:
            st.session_state["agent"] = generate_agent(database, user_id, agent_id, config.config_data['prompt'], config.agent_name)

        load_messages(user_id, agent_id)

    config = configs[selected_prompt_name]

    st.session_state["prior_config_name"] = selected_prompt_name
    agent_id = config.agent_id

    ai_name = st.text_input("Agent Name", config.agent_name, key="agent_name_widget")

    full_prompt = st.text_area(
        "Agent Prompt", config.config_data["prompt"], key="prompt_widget", height=300
    )

    if ai_name != config.agent_name or full_prompt != config.config_data["prompt"]:
        config_data = {
            "prompt": full_prompt,
        }
        agent_config = save_config(
            user_id=user_id,
            agent_name=ai_name,
            config_name=selected_prompt_name,
            config_data=config_data,
            superuser=superuser,
        )
        configs[selected_prompt_name] = agent_config
        st.success(f"Config with name '{selected_prompt_name}' updated successfully!")

    apply_prompt = st.button("Apply", key="apply_widget")
    if apply_prompt:
        st.session_state["agent"] = generate_agent(full_prompt, ai_name)
        memory = st.session_state["agent"].memory
        memory.clear()
        for user, ai in zip(st.session_state.past, st.session_state.generated):
            memory.save_context({"input": user}, {"output": ai})

    elif "agent" not in st.session_state:
        st.session_state["agent"] = generate_agent(full_prompt, ai_name)

    agent = st.session_state["agent"] 
    return agent



def submit():
    st.session_state.submitted_input = st.session_state.input_widget
    st.session_state.input_widget = ""

def main(user_id: str, user_email: str, superuser: bool = False):
    """
    Main function, runs the personal assistant
    """

    # Setup users environement
    create_tables()

    # pull the users config from the database
    configs = load_configs(user_id=user_id, superuser=superuser)

    # Initialize the session state
    init_session_state(configs=configs, superuser=superuser)

    # Create a sidebar
    with st.sidebar:
        agent = render_sidebar(
            configs=configs, superuser=superuser, user_id=user_id
        )

    st.title(f"{agent.ai_prefix} Personal Assistant")

    # pull the users messages from the database
    if st.session_state["loaded_from_db"] == False:
        agent.load_messages()
        st.session_state["loaded_from_db"] = True

    st.text_input(
        "You: ", st.session_state.submitted_input, key="input_widget", on_change=submit
    )

    user_input = st.session_state.submitted_input

    # If the user has submitted input, ask the AI
    if user_input and user_input != st.session_state["last_input"]:
        output = agent.run(user_input)
        st.session_state["last_input"] = user_input

    message_style()
    (generated, past) = agent.messages_to_display()

    # Display the messages
    if generated:
        for i in range(len(generated) - 1, -1, -1):
            message(generated[i], key=str(i))
            message(past[i], is_user=True, key=str(i) + "_user")



if __name__ == "__main__":
    # Set the page title and favicon
    st.set_page_config(page_title=f"GPT Personal Assistant", page_icon=":robot:")

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
        user_id, user_email = login_info
        main(user_id, user_email, superuser=superuser)

    else:
        st.write("Please login")
