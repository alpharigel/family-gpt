import psycopg
import json

from .agent_config import AgentConfig
from .agent_type import StreamlitAgentType

class FamilyGPTDatabaseException(Exception):
    """Exception class for database errors"""

class FamilyGPTDatabaseMessages:
    """Class to handle database operations for storing messages"""

    def __init__(self, db_connection_string: str) -> None:
        """Initialize the class with a connection string"""
        self.db_connection_string = db_connection_string

    def save_message(self, user_id, agent_id, user, ai):
        """Save the mesasage to the database"""

        with psycopg.connect(self.db_connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO messages (user_id, agent_id, message_type, message) VALUES (%s, %s, %s, %s) returning id",
                    (user_id, agent_id, "HUMAN", user),
                )
                rowone = cur.fetchone()
                if rowone is None:
                    raise FamilyGPTDatabaseException("No message saved")
                else:
                    user_message_id = rowone[0]

                cur.execute(
                    "INSERT INTO messages (user_id, agent_id, message_type, message) VALUES (%s, %s, %s, %s) returning id",
                    (user_id, agent_id, "AI", ai),
                )
                rowone = cur.fetchone()
                if rowone is None:
                    raise FamilyGPTDatabaseException("No message saved")
                else:
                    ai_message_id = rowone[0]
                conn.commit()

        return user_message_id, ai_message_id

    def delete_message(self, user_id, agent_id, message_id):
        """Delete a message from the database"""

        with psycopg.connect(self.db_connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM messages WHERE user_id = %s AND agent_id = %s AND id = %s",
                    (user_id, agent_id, message_id),
                )
                conn.commit()

    def save_messages(self, user_id, agent_id, past, generated):
        """Save the mesasages to the database"""
        user_message_id = []
        ai_message_id = []

        with psycopg.connect(self.db_connection_string) as conn:
            with conn.cursor() as cur:
                for user, ai in zip(past, generated):
                    cur.execute(
                        "INSERT INTO messages (user_id, agent_id, message_type, message) VALUES (%s, %s, %s, %s) RETURNING id",
                        (user_id, agent_id, "HUMAN", user),
                    )
                    rowone = cur.fetchone()
                    if rowone is None:
                        raise FamilyGPTDatabaseException("No message saved")
                    user_message_id += rowone[0]

                    cur.execute(
                        "INSERT INTO messages (user_id, agent_id, message_type, message) VALUES (%s, %s, %s, %s) RETURNING id",
                        (user_id, agent_id, "AI", ai),
                    )
                    rowone = cur.fetchone()
                    if rowone is None:
                        raise FamilyGPTDatabaseException("No message saved")
                    ai_message_id += rowone[0]
                    conn.commit()

        return user_message_id, ai_message_id

    def create_tables(self):
        conn = psycopg.connect(self.db_connection_string)
        cur = conn.cursor()

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

    def load_messages(self, user_id: str, agent_id: int):
        with psycopg.connect(self.db_connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT message_type, message, id FROM Messages WHERE user_id = %s and agent_id = %s",
                    (user_id, agent_id),
                )
                messages = cur.fetchall()
                return messages


import os


class FamilyGPTDatabaseAgents:
    """Class to handle database operations"""

    def __init__(self, db_connection_string: str = os.environ["DATABASE_URL"]) -> None:
        """Initialize the class with a connection string"""
        self.db_connection_string = db_connection_string

    def create_tables(self):
        conn = psycopg.connect(self.db_connection_string)
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
                agent_type TEXT NOT NULL,
                hidden BOOLEAN NOT NULL DEFAULT FALSE,
                CONSTRAINT unique_config UNIQUE (user_id, config_name, hidden, agent_type)
                );
            """
        )

        # If agent type isnt a column, add it with default StreamlitAgentType.CONVERSATION_CHAIN value for existing rows
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'agents' AND column_name = 'agent_type';"
        )
        if not cur.fetchone():
            query = f"ALTER TABLE agents ADD COLUMN agent_type TEXT DEFAULT '{StreamlitAgentType.CONVERSATION_CHAIN.value}';"
            cur.execute(query)

            # make sure the (user_id, config_name, hidden, agent_type) is unique in the table by updating the CONSTRAINT
            cur.execute("ALTER TABLE agents DROP CONSTRAINT unique_config;")
            cur.execute(
                "ALTER TABLE agents ADD CONSTRAINT unique_config UNIQUE (user_id, config_name, hidden, agent_type);"
            )

        conn.commit()
        cur.close()
        conn.close()

    def load_configs(
        self, user_id: str, superuser: bool, agent_type: StreamlitAgentType
    ) -> dict[str, AgentConfig]:
        """Load the configs for this user from the database

        Args:
            user_id (str): The user id
            superuser (bool): Whether the user is a superuser
            default_prompt (str): The default prompt to use if no configs are found

        Returns:
            dict(str, AgentConfig): A dictionary of AgentConfig objects
        """
        with psycopg.connect(self.db_connection_string) as conn:
            with conn.cursor() as cur:
                # Select configs for this user from agents table
                query = f"""
                SELECT agent_id, config_name, config_data, update_date, agent_name 
                FROM agents 
                WHERE user_id = '{user_id}' 
                and hidden = {superuser}
                and agent_type = '{agent_type.value}'
                ORDER BY update_date DESC;
                """
                cur.execute(query)  # type: ignore
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
                        agent_type=agent_type,
                    )

        # if no configs for this user, initialize with default config
        if len(config_dict) == 0:
            config_name = "Base"
            agent_name = "AI"
            config_data = dict()
            agent_config = self.save_config(
                user_id,
                config_name,
                config_data,
                superuser,
                agent_name,
                agent_type=agent_type,
            )
            config_dict[config_name] = agent_config

        return config_dict

    def save_config(
        self,
        user_id: str,
        config_name: str,
        config_data: dict,
        superuser: bool,
        agent_name: str,
        agent_type: StreamlitAgentType,
    ) -> AgentConfig:
        with psycopg.connect(self.db_connection_string) as conn:
            with conn.cursor() as cur:
                # Insert config into agents table
                config_json = json.dumps(config_data)
                query = """INSERT INTO Agents 
                (user_id, config_name, config_data, agent_name, hidden, agent_type) 
                VALUES (%s, %s, %s, %s, %s, %s) 
                ON CONFLICT (user_id, config_name, hidden, agent_type) 
                DO UPDATE SET config_data = %s, 
                update_date = CURRENT_TIMESTAMP, 
                agent_name = %s 
                RETURNING agent_id, update_date;
                """
                cur.execute(
                    query,
                    (
                        user_id,
                        config_name,
                        config_json,
                        agent_name,
                        superuser,
                        agent_type.value,
                        config_json,
                        agent_name,
                    ),
                )
                c = cur.fetchone()
                if c is None:
                    raise FamilyGPTDatabaseException("Could not save config")
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
            agent_type=agent_type,
        )
        return agent_config

    def delete_config(
        self,
        user_id: str,
        config_name: str,
        superuser: bool,
        agent_type: StreamlitAgentType,
    ):
        with psycopg.connect(self.db_connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM Agents WHERE user_id = %s AND config_name = %s AND hidden = %s and agent_type = %s",
                    (user_id, config_name, superuser, agent_type.value),
                )
                conn.commit()

    def rename_config(
        self,
        user_id: str,
        config_name: str,
        new_config_name: str,
        superuser: bool,
        agent_type: StreamlitAgentType,
    ):
        with psycopg.connect(self.db_connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE Agents SET config_name = %s WHERE user_id = %s AND config_name = %s AND hidden = %s and agent_type = %s",
                    (
                        new_config_name,
                        user_id,
                        config_name,
                        superuser,
                        agent_type.value,
                    ),
                )
                conn.commit()
