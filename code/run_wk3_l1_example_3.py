# scripts/run_lesson1_example3_terminal.py

import sys
from pathlib import Path
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

sys.path.append(str(Path(__file__).parent.parent))

from utils import load_publication, load_yaml_config, load_env, save_text_to_file
from paths import OUTPUTS_DIR, APP_CONFIG_FPATH


def run_interactive_conversation(publication_content: str, model_name: str) -> None:
    """Runs an interactive terminal-based conversation with the LLM and saves it."""
    # Initialize the LLM
    llm = ChatGroq(
        model=model_name,
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Initialize conversation
    conversation = [
        SystemMessage(content=f"""
You are a helpful AI assistant discussing a research publication.
Base your answers only on this publication content:

{publication_content}
""")
    ]

    print("\nInteractive Q&A Assistant — VAE Publication Chat 📝")
    print("Type your question and press Enter. Type 'q' to quit.\n")

    # Save conversation transcript (as a list of formatted segments)
    transcript_segments = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit" or user_input.lower() == "q":
            print("Exiting. Goodbye!")
            break

        # Append user's message
        conversation.append(HumanMessage(content=user_input))
        transcript_segments.append(
            "======================================================================" + "\n"
            f"👤 YOU:\n\n{user_input.strip()}\n"
            "======================================================================"
        )

        # Get the LLM's response
        response = llm.invoke(conversation)
        print("🤖 AI Response:\n\n" + response.content + "\n")

        # Append AI's response to the conversation history
        conversation.append(AIMessage(content=response.content))
        transcript_segments.append(
            "======================================================================" + "\n"
            f"🤖 AI Response:\n\n{response.content.strip()}\n"
            "======================================================================"
        )
        print("=" * 60)

    # Save transcript to a file with a clear header
    transcript = (
        "📝 **Transcript of Example 3: Interactive Conversation with LLM**\n\n" +
        "\n\n".join(transcript_segments)
    )
    save_text_to_file(
        transcript,
        os.path.join(OUTPUTS_DIR, "example3_terminal_conversation.md"),
        header="Example 3: Interactive Conversation with LLM"
    )
    print("✓ Conversation transcript saved.")


def main() -> None:
    """Main entry point to run Example 3 from Lesson 1 (interactive terminal chat)."""
    try:
        print("=" * 80)
        print("\nLoading environment variables...")
        load_env()
        print("✓ Groq API key loaded.")

        print("Loading publication content...")
        vae_publication_id = 'yzN0OCQT7hUS'
        publication_content = load_publication(publication_external_id=vae_publication_id)
        print(f"✓ Publication loaded ({len(publication_content)} characters).")

        print("Loading application configuration...")
        app_config = load_yaml_config(APP_CONFIG_FPATH)
        model_name = app_config.get("llm", "llama-3.1-8b-instant")
        print(f"✓ Model set to: {model_name}")

        run_interactive_conversation(publication_content, model_name)

        print("\n" + "-"*80)
        print("TASK COMPLETE!")
        print("=" * 80)

    except Exception as e:
        print(f"Error in script execution: {e}")
        return None


if __name__ == "__main__":
    main()
