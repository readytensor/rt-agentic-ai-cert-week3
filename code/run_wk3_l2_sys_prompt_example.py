import sys
from pathlib import Path
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

sys.path.append(str(Path(__file__).parent.parent))

from utils import load_publication, load_yaml_config, load_env, save_text_to_file
from prompt_builder import build_system_prompt_from_config, print_prompt_preview
from paths import OUTPUTS_DIR, APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH


def clear_screen():
    if os.name == "nt":  # Windows
        os.system("cls")
    else:
        os.system("clear")


def print_instructions(config_name: str):
    print("\n" + "="*80)
    print(f"Interactive Q&A Assistant with System Prompt — VAE Publication Chat 📝")
    print(f"Using config: {config_name}")
    print("Type your question and press Enter.")
    print("Type 'q' to quit or 'c' to clear the screen.\n")


def run_interactive_conversation_with_system_prompt(
    publication_content: str, 
    model_name: str,
    system_prompt_config_name: str = "ai_assistant_system_prompt_professional"
) -> None:
    """Runs an interactive terminal-based conversation using a configured system prompt."""
    
    # Load prompt configurations
    prompt_configs = load_yaml_config(PROMPT_CONFIG_FPATH)
    system_prompt_config = prompt_configs.get(system_prompt_config_name)
    
    if not system_prompt_config:
        raise ValueError(f"System prompt config '{system_prompt_config_name}' not found")
    
    # Build the system prompt
    system_prompt = build_system_prompt_from_config(
        system_prompt_config, 
        publication_content
    )
    
    print("\n" + "="*80)
    print("SYSTEM PROMPT PREVIEW:")
    print("="*80)
    print_prompt_preview(system_prompt, max_length=800)
    print("\n")
    
    # Initialize the LLM
    llm = ChatGroq(
        model=model_name,
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Initialize conversation with the built system prompt
    conversation = [SystemMessage(content=system_prompt)]

    print(f"\nInteractive Q&A Assistant with System Prompt — VAE Publication Chat 📝")
    print(f"Using config: {system_prompt_config_name}")
    print("Type your question and press Enter. Type 'q' to quit.\n")

    # Save conversation transcript - now including the system prompt
    transcript_segments = [
        f"System Prompt Config: {system_prompt_config_name}\n"
        f"Description: {system_prompt_config.get('description', 'N/A')}\n"
        "======================================================================\n"
        "📋 **CONSTRUCTED SYSTEM PROMPT:**\n"
        "======================================================================\n"
        f"{system_prompt}\n"
        "======================================================================\n"
        "💬 **CONVERSATION:**\n"
        "======================================================================"
    ]

    while True:
        print_instructions(system_prompt_config_name)
        user_input = input("You: ")
        
        # Check if user wants to clear the screen
        if user_input.lower() in ["c", "clear"]:
            clear_screen()
            continue
        
        if user_input.lower() in ["quit", "q"]:
            print("Exiting. Goodbye!")
            break

        # Append user's message
        conversation.append(HumanMessage(content=user_input))
        transcript_segments.append(
            "======================================================================" + "\n"
            f"👤 YOU:\n\n{user_input.strip()}\n"
            "======================================================================"
        )

        try:
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
        except Exception as e:
            print(f"Error getting response: {e}")
            
        print("=" * 60)

    # Save transcript to a file
    transcript = (
        "📝 **Transcript: System Prompt Enhanced Conversation**\n\n" +
        "\n\n".join(transcript_segments)
    )
    save_text_to_file(
        transcript,
        os.path.join(OUTPUTS_DIR, f"lesson2_system_prompt_{system_prompt_config_name}.md"),
        header=f"Lesson 2: System Prompt Example - {system_prompt_config_name}"
    )
    print("✓ Conversation transcript saved.")


def main() -> None:
    """Main entry point for system prompt demonstration."""
    try:
        print("=" * 80)
        print("LESSON 2: SYSTEM PROMPT DEMONSTRATION")
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

        # Ask user which system prompt config to use
        print("\nAvailable system prompt configurations:")
        print("1. ai_assistant_system_prompt_basic")
        print("2. ai_assistant_system_prompt_advanced")
        
        choice = input("\nSelect configuration (1 or 2, default=2): ").strip()
        
        if choice == "1":
            config_name = "ai_assistant_system_prompt_basic"
        else:
            config_name = "ai_assistant_system_prompt_advanced"
            
        run_interactive_conversation_with_system_prompt(
            publication_content, 
            model_name,
            config_name
        )

        print("\n" + "-"*80)
        print("TASK COMPLETE!")
        print("=" * 80)

    except Exception as e:
        print(f"Error in script execution: {e}")
        return None


if __name__ == "__main__":
    main()