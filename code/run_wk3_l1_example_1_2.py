# scripts/run_lesson1_examples.py

import sys
from pathlib import Path
import os
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.append(str(Path(__file__).parent.parent))

from utils import load_publication, load_yaml_config, load_env, save_text_to_file
from paths import OUTPUTS_DIR, APP_CONFIG_FPATH


def invoke_llm(messages: list, model: str = "llama-3.1-8b-instant", temperature: float = 0.7) -> Optional[str]:
    """Calls the LLM with a list of messages and returns the response content."""
    try:
        llm = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY")
        )
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None


def run_example1(model_name: str) -> None:
    """Example 1: General question about VAEs (no context)."""
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content="What are variational autoencoders and list the top 5 applications for them?")
    ]
    response = invoke_llm(messages, model=model_name)
    if response:
        save_text_to_file(
            response,
            os.path.join(OUTPUTS_DIR, "example1_llm_response.md"),
            header="Example 1: LLM Response (General VAE Question)"
        )
        print("✓ Example 1 complete.")
    else:
        print("✗ Example 1 failed.")


def run_example2(publication_content: str, model_name: str) -> None:
    """Example 2: Same question but grounded in publication content."""
    system_prompt = "You are a helpful AI assistant."
    user_question = f"""
Based on this publication: {publication_content}

What are variational autoencoders and list the top 5 applications for them as discussed in this publication.
"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ]
    response = invoke_llm(messages, model=model_name)
    if response:
        save_text_to_file(
            response,
            os.path.join(OUTPUTS_DIR, "example2_llm_response.md"),
            header="Example 2: LLM Response (Publication-Specific Answer)"
        )
        print("✓ Example 2 complete.")
    else:
        print("✗ Example 2 failed.")


def main() -> None:
    """Main entry point to run Examples 1 and 2 from Lesson 1."""
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

        print("\nRunning Example 1: General knowledge response...")
        run_example1(model_name)

        print("\nRunning Example 2: Grounded response from publication...")
        run_example2(publication_content, model_name)

        print("\n" + "-"*80)
        print("TASK COMPLETE!")
        print("=" * 80)

    except Exception as e:
        print(f"Error in script execution: {e}")
        return None


if __name__ == "__main__":
    main()
