import os
import sqlite3
import warnings
from datetime import datetime
from paths import CHAT_HISTORY_DB_FPATH, APP_CONFIG_FPATH
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from utils import load_env, load_yaml_config

warnings.filterwarnings("ignore")


class ChatWithMemory:
    """Simple chat with persistent memory."""

    def __init__(self):
        """Initialize the chat."""
        load_env()

        # Load config and setup LLM
        app_config = load_yaml_config(APP_CONFIG_FPATH)
        model_name = app_config.get("llm", "llama-3.1-8b-instant")

        self.llm = ChatGroq(
            model=model_name, temperature=0.7, api_key=os.getenv("GROQ_API_KEY")
        )

        self.current_session = None
        self.memory = None
        self.chat_history = []  # Cache chat history in memory

    def start_session(self, session_name: str = None):
        """Start or load a chat session."""
        if not session_name:
            session_name = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_session = session_name

        # Setup memory
        history = SQLChatMessageHistory(
            connection=f"sqlite:///{CHAT_HISTORY_DB_FPATH}",
            session_id=session_name,
        )

        # Or use PostgresChatMessageHistory
        # history = PostgresChatMessageHistory(
        #     connection_string=f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}",
        #     session_id=session_name,
        # )

        # Or use FileChatMessageHistory
        # history = FileChatMessageHistory(
        #     file_path=f"{CHAT_HISTORY_DB_FPATH}",
        #     session_id=session_name,
        # )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history", chat_memory=history, return_messages=True
        )

        # Load chat history once when starting session
        memory_vars = self.memory.load_memory_variables({})
        self.chat_history = memory_vars.get("chat_history", [])

        # Check if existing session
        existing_messages = len(self.chat_history)
        if existing_messages > 0:
            print(
                f"Loaded existing session '{session_name}' with {existing_messages} messages"
            )
        else:
            print(f"Started new session '{session_name}'")

    def ask(self, user_input: str) -> str:
        """Send message and get response."""
        if not self.memory:
            raise ValueError("No active session. Call start_session() first.")

        # Build messages using cached chat history
        messages = [SystemMessage(content="You are a helpful AI assistant.")]
        messages.extend(self.chat_history)
        messages.append(HumanMessage(content=user_input))

        response = self.llm.invoke(messages)

        # Save to persistent memory
        self.memory.save_context({"input": user_input}, {"output": response.content})

        # Update cached chat history
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(response)

        return response.content

    def list_sessions(self):
        """List all sessions."""
        try:
            conn = sqlite3.connect(CHAT_HISTORY_DB_FPATH)
            cursor = conn.cursor()

            # Create table if needed
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS message_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute(
                "SELECT DISTINCT session_id FROM message_store ORDER BY session_id"
            )
            sessions = [row[0] for row in cursor.fetchall()]
            conn.close()
            return sessions
        except:
            return []

    def get_session_messages(self, session_id: str) -> list:
        """Get all messages from a specific session."""
        try:
            # Create temporary history object for the session
            temp_history = SQLChatMessageHistory(
                connection=f"sqlite:///{CHAT_HISTORY_DB_FPATH}",
                session_id=session_id,
            )
            return temp_history.messages
        except Exception as e:
            print(f"Error getting messages for session {session_id}: {e}")
            return []

    def display_session_messages(self, session_id: str, max_messages: int = None):
        """Display messages from a session in a readable format."""
        messages = self.get_session_messages(session_id)

        if not messages:
            print(f"No messages found in session: {session_id}")
            return

        print(f"\n Messages in session: {session_id}")
        print("=" * 50)

        # Limit messages if specified
        if max_messages and len(messages) > max_messages:
            print(f"Showing last {max_messages} of {len(messages)} messages:")
            messages = messages[-max_messages:]
        else:
            print(f"Total messages: {len(messages)}")

        print("-" * 50)

        for i, msg in enumerate(messages, 1):
            # Determine message type
            if hasattr(msg, "type"):
                msg_type = "👤 You" if msg.type == "human" else "🤖 AI"
            else:
                msg_type = "❓ Unknown"

            # Format content
            content = msg.content.strip()
            if len(content) > 200:
                content = content[:200] + "..."

            print(f"{i:2d}. {msg_type}: {content}")

            # Add spacing between messages
            if i < len(messages):
                print()


def main():
    print("🤖 AI Chat with Persistent Memory")
    print("=" * 40)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(CHAT_HISTORY_DB_FPATH), exist_ok=True)

    chat = ChatWithMemory()

    # Show existing sessions
    sessions = chat.list_sessions()
    if sessions:
        print(f"\nExisting sessions: {', '.join(sessions)}")

    # Get session name
    session_name = input("\nEnter session name (or press Enter for new): ").strip()
    if not session_name:
        session_name = None

    # Start session
    chat.start_session(session_name)

    print(f"\n💬 Chatting in session: {chat.current_session}")
    print("Commands:")
    print("  'quit' - exit")
    print("  'sessions' - list all sessions")
    print("  'history' - show current session messages")
    print("  'view <session_name>' - show messages from specific session")
    print("-" * 40)

    # Chat loop
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "quit":
                print("Goodbye! 👋")
                break
            elif user_input.lower() == "sessions":
                sessions = chat.list_sessions()
                print(f"All sessions: {', '.join(sessions) if sessions else 'None'}")
                continue
            elif user_input.lower() == "history":
                chat.display_session_messages(chat.current_session)
                continue
            elif user_input.lower().startswith("view "):
                session_to_view = user_input[5:].strip()
                if session_to_view:
                    chat.display_session_messages(session_to_view, max_messages=10)
                else:
                    print("Usage: view <session_name>")
                continue
            elif user_input:
                response = chat.ask(user_input)
                print(f"AI: {response}")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
