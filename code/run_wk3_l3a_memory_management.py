import sys
from pathlib import Path
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# Removed deprecated LangChain memory imports - implementing custom memory management
import tiktoken

sys.path.append(str(Path(__file__).parent.parent))

from utils import load_publication, load_yaml_config, load_env, save_text_to_file
from prompt_builder import build_system_prompt_from_config
from paths import OUTPUTS_DIR, APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, DATA_DIR


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback estimation if tiktoken fails
        return len(text.split()) * 1.3


def apply_trimming_strategy(conversation_history: list, window_size: int, system_prompt: str) -> list:
    """Apply trimming strategy - keep only recent messages."""
    # Always keep system prompt + last N user/AI message pairs
    system_msg = [SystemMessage(content=system_prompt)]
    
    if len(conversation_history) <= window_size:
        return system_msg + conversation_history
    else:
        return system_msg + conversation_history[-window_size:]


def messages_to_string(messages: list) -> str:
    """Convert message list to string for token counting."""
    content = ""
    for msg in messages:
        if hasattr(msg, 'content'):
            content += f"{msg.__class__.__name__}: {msg.content}\n"
        else:
            content += str(msg) + "\n"
    return content


def run_memory_strategy_comparison(
    publication_content: str,
    questions: list,
    model_name: str,
    app_config: dict,
    system_prompt_config_name: str = "ai_assistant_system_prompt_advanced"
) -> None:
    """Compare three memory strategies: stuffing, trimming, and summarization."""
    
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
    
    # Initialize the LLM
    llm = ChatGroq(
        model=model_name,
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    print(f"\n🧠 Memory Strategy Comparison — VAE Publication Chat")
    print(f"System Prompt Config: {system_prompt_config_name}")
    print(f"Total questions to process: {len(questions)}")
    print("=" * 80)
    
    # Capture final prompts for comparison
    final_stuffing_prompt = ""
    final_trimming_prompt = ""
    final_summary_prompt = ""
    
    # Strategy 1: Stuffing (keep everything)
    print("\n📝 Strategy 1: STUFFING (Keep All Messages)")
    print("-" * 50)
    
    stuffing_conversation = [SystemMessage(content=system_prompt)]
    stuffing_tokens = []
    stuffing_responses = []
    
    for i, question in enumerate(questions, 1):
        # Add user message
        stuffing_conversation.append(HumanMessage(content=question))
        
        # Count tokens before LLM call
        prompt_tokens = count_tokens(messages_to_string(stuffing_conversation))
        
        # Capture final prompt for comparison
        if i == len(questions):
            final_stuffing_prompt = messages_to_string(stuffing_conversation)
        
        try:
            # Get response
            response = llm.invoke(stuffing_conversation)
            response_tokens = count_tokens(response.content)
            
            # Add AI response to conversation
            stuffing_conversation.append(AIMessage(content=response.content))
            
            # Track metrics
            stuffing_tokens.append({
                'question_num': i,
                'prompt_tokens': prompt_tokens,
                'response_tokens': response_tokens,
                'total_tokens': prompt_tokens + response_tokens
            })
            stuffing_responses.append({
                'question': question,
                'response': response.content
            })
            
            if i % 10 == 0:
                print(f"  ✓ Processed {i} questions, current prompt: {prompt_tokens} tokens")
                
        except Exception as e:
            print(f"  ❌ Error at question {i}: {e}")
            break
    
    # Get memory strategy configurations
    memory_config = app_config.get("memory_strategies", {})
    trimming_window_size = memory_config.get("trimming_window_size", 8)
    summarization_max_tokens = memory_config.get("summarization_max_tokens", 1000)
    
    # Strategy 2: Trimming (keep last N messages)
    print(f"\n✂️ Strategy 2: TRIMMING (Keep Last {trimming_window_size} Messages)")
    print("-" * 50)
    
    trimming_conversation = []
    trimming_tokens = []
    trimming_responses = []
    
    for i, question in enumerate(questions, 1):
        # Add user message to history
        trimming_conversation.append(HumanMessage(content=question))
        
        # Apply trimming strategy
        messages = apply_trimming_strategy(trimming_conversation[:-1], trimming_window_size, system_prompt)
        messages.append(HumanMessage(content=question))
        
        # Count tokens
        prompt_tokens = count_tokens(messages_to_string(messages))
        
        # Capture final prompt for comparison
        if i == len(questions):
            final_summary_prompt = messages_to_string(messages)
        
        # Capture final prompt for comparison
        if i == len(questions):
            final_trimming_prompt = messages_to_string(messages)
        
        try:
            # Get response
            response = llm.invoke(messages)
            response_tokens = count_tokens(response.content)
            
            # Add AI response to conversation history
            trimming_conversation.append(AIMessage(content=response.content))
            
            # Track metrics
            trimming_tokens.append({
                'question_num': i,
                'prompt_tokens': prompt_tokens,
                'response_tokens': response_tokens,
                'total_tokens': prompt_tokens + response_tokens
            })
            trimming_responses.append({
                'question': question,
                'response': response.content
            })
            
            if i % 10 == 0:
                print(f"  ✓ Processed {i} questions, current prompt: {prompt_tokens} tokens")
                
        except Exception as e:
            print(f"  ❌ Error at question {i}: {e}")
            break
    
    # Strategy 3: Summarization (compress old conversations)
    print(f"\n📋 Strategy 3: SUMMARIZATION (Compress History, max {summarization_max_tokens} tokens)")
    print("-" * 50)
    
    summary_conversation = []
    summary_tokens = []
    summary_responses = []
    
    for i, question in enumerate(questions, 1):
        # Add user message to history
        summary_conversation.append(HumanMessage(content=question))
        
        # Apply summarization strategy
        messages = apply_summarization_strategy(
            summary_conversation[:-1], 
            llm, 
            summarization_max_tokens, 
            system_prompt
        )
        messages.append(HumanMessage(content=question))
        
        # Count tokens
        prompt_tokens = count_tokens(messages_to_string(messages))
        
        try:
            # Get response
            response = llm.invoke(messages)
            response_tokens = count_tokens(response.content)
            
            # Add AI response to conversation history
            summary_conversation.append(AIMessage(content=response.content))
            
            # Track metrics
            summary_tokens.append({
                'question_num': i,
                'prompt_tokens': prompt_tokens,
                'response_tokens': response_tokens,
                'total_tokens': prompt_tokens + response_tokens
            })
            summary_responses.append({
                'question': question,
                'response': response.content
            })
            
            if i % 10 == 0:
                print(f"  ✓ Processed {i} questions, current prompt: {prompt_tokens} tokens")
                
        except Exception as e:
            print(f"  ❌ Error at question {i}: {e}")
            break
    
    # Generate comparison report
    generate_comparison_report(
        stuffing_tokens, trimming_tokens, summary_tokens,
        stuffing_responses, trimming_responses, summary_responses,
        system_prompt, questions,
        final_stuffing_prompt, final_trimming_prompt, final_summary_prompt
    )


def generate_comparison_report(
    stuffing_tokens, trimming_tokens, summary_tokens,
    stuffing_responses, trimming_responses, summary_responses,
    system_prompt, questions,
    final_stuffing_prompt, final_trimming_prompt, final_summary_prompt
):
    """Generate detailed comparison report."""
    
    print("\n" + "=" * 80)
    print("📊 MEMORY STRATEGY COMPARISON RESULTS")
    print("=" * 80)
    
    # Calculate totals
    stuffing_total_prompt = sum(t['prompt_tokens'] for t in stuffing_tokens)
    stuffing_total_response = sum(t['response_tokens'] for t in stuffing_tokens)
    stuffing_total = stuffing_total_prompt + stuffing_total_response
    
    trimming_total_prompt = sum(t['prompt_tokens'] for t in trimming_tokens)
    trimming_total_response = sum(t['response_tokens'] for t in trimming_tokens)
    trimming_total = trimming_total_prompt + trimming_total_response
    
    summary_total_prompt = sum(t['prompt_tokens'] for t in summary_tokens)
    summary_total_response = sum(t['response_tokens'] for t in summary_tokens)
    summary_total = summary_total_prompt + summary_total_response
    
    # Print summary table
    print(f"\n{'Strategy':<15} {'Prompt Tokens':<15} {'Response Tokens':<15} {'Total Tokens':<15}")
    print("-" * 65)
    print(f"{'Stuffing':<15} {stuffing_total_prompt:<15,} {stuffing_total_response:<15,} {stuffing_total:<15,}")
    print(f"{'Trimming':<15} {trimming_total_prompt:<15,} {trimming_total_response:<15,} {trimming_total:<15,}")
    print(f"{'Summarization':<15} {summary_total_prompt:<15,} {summary_total_response:<15,} {summary_total:<15,}")
    
    # Show efficiency gains
    print(f"\n📈 Efficiency Comparison (vs Stuffing):")
    trimming_savings = ((stuffing_total - trimming_total) / stuffing_total) * 100
    summary_savings = ((stuffing_total - summary_total) / stuffing_total) * 100
    print(f"  Trimming saves: {trimming_savings:.1f}% tokens")
    print(f"  Summarization saves: {summary_savings:.1f}% tokens")
    
    # Generate detailed report file
    report_content = []
    report_content.append("# Memory Strategy Comparison Report")
    report_content.append("=" * 50)
    report_content.append("")
    
    # Add summary table
    report_content.append("## Token Usage Summary")
    report_content.append("")
    report_content.append(f"| Strategy | Prompt Tokens | Response Tokens | Total Tokens |")
    report_content.append(f"|----------|---------------|-----------------|--------------|")
    report_content.append(f"| Stuffing | {stuffing_total_prompt:,} | {stuffing_total_response:,} | {stuffing_total:,} |")
    report_content.append(f"| Trimming | {trimming_total_prompt:,} | {trimming_total_response:,} | {trimming_total:,} |")
    report_content.append(f"| Summarization | {summary_total_prompt:,} | {summary_total_response:,} | {summary_total:,} |")
    report_content.append("")
    
    # Add final prompt examples
    report_content.append("## Final Prompt Comparison")
    report_content.append("")
    report_content.append("### System Prompt (Same for All)")
    report_content.append("```")
    report_content.append(system_prompt[:500] + "..." if len(system_prompt) > 500 else system_prompt)
    report_content.append("```")
    report_content.append("")
    
    # Show what the final question's prompt looked like for each strategy
    if len(questions) > 0:
        final_question = questions[-1]
        report_content.append(f"### Final Question: '{final_question}'")
        report_content.append("")
        
        report_content.append("**Stuffing Strategy Final Prompt:**")
        report_content.append(f"- Contains ALL {len(stuffing_responses)} previous Q&A pairs")
        report_content.append(f"- Total prompt size: {stuffing_tokens[-1]['prompt_tokens'] if stuffing_tokens else 0:,} tokens")
        report_content.append("")
        report_content.append("```")
        report_content.append(final_stuffing_prompt[:1000] + "..." if len(final_stuffing_prompt) > 1000 else final_stuffing_prompt)
        report_content.append("```")
        report_content.append("")
        
        report_content.append("**Trimming Strategy Final Prompt:**")
        report_content.append("- Contains last 8 messages only (4 Q&A pairs)")
        report_content.append(f"- Total prompt size: {trimming_tokens[-1]['prompt_tokens'] if trimming_tokens else 0:,} tokens")
        report_content.append("")
        report_content.append("```")
        report_content.append(final_trimming_prompt[:1000] + "..." if len(final_trimming_prompt) > 1000 else final_trimming_prompt)
        report_content.append("```")
        report_content.append("")
        
        report_content.append("**Summarization Strategy Final Prompt:**")
        report_content.append("- Contains summary + recent messages")
        report_content.append(f"- Total prompt size: {summary_tokens[-1]['prompt_tokens'] if summary_tokens else 0:,} tokens")
        report_content.append("")
        report_content.append("```")
        report_content.append(final_summary_prompt[:1000] + "..." if len(final_summary_prompt) > 1000 else final_summary_prompt)
        report_content.append("```")
        report_content.append("")
    
    # Save report
    report_text = "\n".join(report_content)
    save_text_to_file(
        report_text,
        os.path.join(OUTPUTS_DIR, "lesson3a_memory_strategy_comparison.md"),
        header="Lesson 3A: Memory Strategy Comparison"
    )
    
    print(f"\n✓ Detailed comparison report saved to outputs/")


def main() -> None:
    """Main entry point for memory strategy demonstration."""
    try:
        print("=" * 80)
        print("LESSON 3A: MEMORY STRATEGY COMPARISON")
        print("=" * 80)
        
        print("\nLoading environment variables...")
        load_env()
        print("✓ Groq API key loaded.")

        print("Loading publication content...")
        vae_publication_id = 'yzN0OCQT7hUS'
        publication_content = load_publication(publication_external_id=vae_publication_id)
        print(f"✓ Publication loaded ({len(publication_content)} characters).")
        
        print("Loading sample questions...")
        questions_config = load_yaml_config(os.path.join(DATA_DIR, f"{vae_publication_id}-sample-questions.yaml"))
        questions = questions_config.get("questions", [])
        print(f"✓ {len(questions)} questions loaded.")

        print("Loading application configuration...")
        app_config = load_yaml_config(APP_CONFIG_FPATH)
        model_name = app_config.get("llm", "llama-3.1-8b-instant")
        
        # Display memory strategy configs
        memory_config = app_config.get("memory_strategies", {})
        trimming_window = memory_config.get("trimming_window_size", 8)
        summarization_max = memory_config.get("summarization_max_tokens", 1000)
        
        print(f"✓ Model set to: {model_name}")
        print(f"✓ Trimming window size: {trimming_window} messages")
        print(f"✓ Summarization max tokens: {summarization_max}")

        # Ask user how many questions to process
        print(f"\nAvailable questions: {len(questions)}")
        num_questions = input(f"How many questions to process? (default=20, max={len(questions)}): ").strip()
        
        try:
            num_questions = int(num_questions) if num_questions else 20
            num_questions = min(num_questions, len(questions))
        except ValueError:
            num_questions = 20
            
        selected_questions = questions[:num_questions]
        print(f"✓ Will process {len(selected_questions)} questions.")
            
        run_memory_strategy_comparison(
            publication_content, 
            selected_questions,
            model_name,
            app_config  # Pass full config to access memory settings
        )

        print("\n" + "-"*80)
        print("MEMORY STRATEGY COMPARISON COMPLETE!")
        print("=" * 80)

    except Exception as e:
        print(f"Error in script execution: {e}")
        return None


if __name__ == "__main__":
    main()