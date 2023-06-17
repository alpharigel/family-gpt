import streamlit as st
from util.message import message, message_style

import requests
import datetime
import pickle
from dotenv import load_dotenv
import streamlit_google_oauth as oauth
import os

load_dotenv()

client_id = os.environ["GOOGLE_CLIENT_ID"]
client_secret = os.environ["GOOGLE_CLIENT_SECRET"]
redirect_uri = os.environ["GOOGLE_REDIRECT_URI"]
db_connection_string = os.environ["DATABASE_URL"]

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

from util.chat_agent.prompt import BASE_PROMPT


def generate_chatter(user_prompt: str, ai_name: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(user_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )

    llm = ChatOpenAI(temperature=0.8)
    memory = ConversationSummaryBufferMemory(
        llm=llm, max_token_limit=800, return_messages=True, ai_prefix=ai_name
    )
    # memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    return conversation


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


def rebuild_memory():
    memory = st.session_state["chatter"].memory
    memory.clear()
    for user, ai in zip(st.session_state.past, st.session_state.generated):
        memory.save_context({"input": user}, {"output": ai})


import copy

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """
    Data class that contains the minimum data in an agent configuration
    """

    agent_id: str
    agent_name: str
    config_name: str
    config_data: dict  # usually contains 'prompt'
    update_date: datetime.datetime
    hidden: bool


def render_sidebar(configs: dict, user_id: str, superuser: bool = False):
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
        new_config = save_config(
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
        if "chatter" not in st.session_state:
            st.session_state["chatter"] = generate_chatter(config.config_data['prompt'], config.agent_name)

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
        st.session_state["chatter"] = generate_chatter(full_prompt, ai_name)
        memory = st.session_state["chatter"].memory
        memory.clear()
        for user, ai in zip(st.session_state.past, st.session_state.generated):
            memory.save_context({"input": user}, {"output": ai})

    elif "chatter" not in st.session_state:
        st.session_state["chatter"] = generate_chatter(full_prompt, ai_name)

    if st.button("Current summary"):
        memory = st.session_state["chatter"].memory
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
        rebuild_memory()

    if st.button("Load messages"):
        load_messages(user_id=user_id, agent_id=agent_id)

    pop_last = st.button("Undo last message", key="undo_widget")
    if pop_last:
        st.session_state.submitted_input = ""
        st.session_state.past.pop()
        st.session_state.generated.pop()
        delete_message(user_id, agent_id, st.session_state["past_id"].pop())
        delete_message(user_id, agent_id, st.session_state["generated_id"].pop())
        rebuild_memory()
        st.session_state["last_input"] = ""



    return (ai_name, agent_id)

def delete_message(user_id, agent_id, message_id):
    """Delete a message from the database"""

    with psycopg.connect(db_connection_string) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM messages WHERE user_id = %s AND agent_id = %s AND id = %s",
                (user_id, agent_id, message_id),
            )
            conn.commit()

def save_messages(user_id, agent_id, past, generated):
    """Save the mesasages to the database"""

    with psycopg.connect(db_connection_string) as conn:
        with conn.cursor() as cur:
            for user, ai in zip(past, generated):
                cur.execute(
                    "INSERT INTO messages (user_id, agent_id, message_type, message) VALUES (%s, %s, %s, %s) RETURNING id",
                    (user_id, agent_id, "HUMAN", user),
                    )
                user_message_id = cur.fetchone()[0]

                cur.execute(
                    "INSERT INTO messages (user_id, agent_id, message_type, message) VALUES (%s, %s, %s, %s) RETURNING id",
                    (user_id, agent_id, "AI", ai),
                    )
                ai_message_id = cur.fetchone()[0]
                conn.commit()

    return user_message_id, ai_message_id

def save_message(user_id, agent_id, user, ai):
    """Save the mesasage to the database"""

    with psycopg.connect(db_connection_string) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (user_id, agent_id, message_type, message) VALUES (%s, %s, %s, %s) returning id",
                (user_id, agent_id, "HUMAN", user),
                )
            user_message_id = cur.fetchone()[0]

            cur.execute(
                "INSERT INTO messages (user_id, agent_id, message_type, message) VALUES (%s, %s, %s, %s) returning id",
                (user_id, agent_id, "AI", ai),
                )
            ai_message_id = cur.fetchone()[0]
            conn.commit()

    return user_message_id, ai_message_id



def submit():
    st.session_state.submitted_input = st.session_state.input_widget
    st.session_state.input_widget = ""


def ask_chatter(chatter, user_input, past=[], generated=[]):
    response = chatter.predict(input=user_input)
    return response


import psycopg


def create_tables():
    conn = psycopg.connect(db_connection_string)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Agents (
            agent_id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            update_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            agent_name TEXT NOT NULL,
            config_name TEXT NOT NULL,
            config_data JSONB NOT NULL,
            hidden BOOLEAN NOT NULL DEFAULT FALSE,
            CONSTRAINT unique_config UNIQUE (user_id, config_name, hidden)
            );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Messages (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        user_id VARCHAR(255) NOT NULL,
        agent_id INTEGER NOT NULL,
        message_type VARCHAR(255) NOT NULL,
        message TEXT NOT NULL
        );
        """
    )

    conn.commit()
    cur.close()
    conn.close()


def load_configs(
    user_id: str, superuser: bool = False, default_prompt: str = BASE_PROMPT
):
    with psycopg.connect(db_connection_string) as conn:
        with conn.cursor() as cur:
            # Select configs for this user from agents table
            query = f"SELECT agent_id, config_name, config_data, update_date, agent_name FROM Agents WHERE user_id = '{user_id}' and hidden = {superuser} ORDER BY update_date DESC;"
            cur.execute(query)
            configs = cur.fetchall()

            # Shape into dictonary for returning to user
            config_dict = {}
            for c in configs:
                config_dict[c[1]] = AgentConfig(
                    agent_id=c[0],
                    config_name=c[1],
                    config_data=c[2],
                    update_date=c[3],
                    agent_name=c[4],
                    hidden=superuser,
                )

    # if no configs for this user, initialize with default config
    if len(config_dict) == 0:
        config_name = "Base"
        agent_name = "AI"
        config_data = {
            "prompt": default_prompt,
        }
        agent_config = save_config(user_id, config_name, config_data, superuser, agent_name)
        config_dict[config_name] = agent_config

    return config_dict


import json


def save_config(user_id: str, config_name: str, config_data: dict, superuser: bool, agent_name: str):
    with psycopg.connect(db_connection_string) as conn:
        with conn.cursor() as cur:
            # Insert config into agents table
            config_json = json.dumps(config_data)
            query = "INSERT INTO Agents (user_id, config_name, config_data, agent_name, hidden) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (user_id, config_name, hidden) DO UPDATE SET config_data = %s, update_date = CURRENT_TIMESTAMP, agent_name = %s RETURNING agent_id, update_date;"
            cur.execute(
                query,
                (
                    user_id,
                    config_name,
                    config_json,
                    agent_name,
                    superuser,
                    config_json,
                    agent_name,
                ),
            )
            c = cur.fetchone()
            agent_id = c[0]
            update_date = c[1]
            conn.commit()

    agent_config = AgentConfig(
        agent_id=agent_id,
        config_name=config_name,
        config_data=config_data,
        update_date=update_date,
        agent_name=agent_name,
        hidden=superuser,
    )
    return agent_config


def load_messages(user_id: str, agent_id: int):
    with psycopg.connect(db_connection_string) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT message_type, message, id FROM Messages WHERE user_id = %s and agent_id = %s", (user_id,agent_id))
            messages = cur.fetchall()
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
            rebuild_memory()


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
        (ai_name, agent_id) = render_sidebar(
            configs=configs, superuser=superuser, user_id=user_id
        )

    st.title(f"{ai_name} Personal Assistant")

    # pull the users messages from the database
    if st.session_state["loaded_from_db"] == False:
        load_messages(user_id=user_id, agent_id=agent_id)
        st.session_state["loaded_from_db"] = True

    st.text_input(
        "You: ", st.session_state.submitted_input, key="input_widget", on_change=submit
    )

    user_input = st.session_state.submitted_input

    # If the user has submitted input, ask the AI
    if user_input and user_input != st.session_state["last_input"]:
        output = ask_chatter(
            st.session_state["chatter"],
            user_input,
            past=st.session_state["past"],
            generated=st.session_state["generated"],
        )

        # Save the message to the database
        
        (user_message_id, ai_message_id) = save_message(user_id=user_id, agent_id=agent_id, user=user_input, ai=output)

        st.session_state.past.append(user_input)
        st.session_state.past_id.append(user_message_id)
        st.session_state.generated.append(output)
        st.session_state.generated_id.append(ai_message_id)
        st.session_state["last_input"] = user_input

    message_style()

    # Display the messages
    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")



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
