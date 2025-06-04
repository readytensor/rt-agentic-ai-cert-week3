import sys
from pathlib import Path
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import tiktoken

sys.path.append(str(Path(__file__).parent.parent))

from utils import load_publication, load_yaml_config, load_env, save_text_to_file
from prompt_builder import build_system_prompt_from_config
from paths import OUTPUTS_DIR, APP_CONFIG_FPATH, PROMPT_CONFIG_FPATH, DATA_DIR


def count_tokens(text: str, model: str = "gpt-3.5-turbo"):
    """Count tokens using tiktoken with fallback."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback estimation
        return int(len(text.split()) * 1.3)


def messages_to_string(messages: list, include_publication: bool = False) -> str:
    """Convert message list to readable string."""
    content = ""
    user_question_count = 0
    
    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            system_content = msg.content
            # Remove publication content if requested
            if not include_publication and "=== PUBLICATION CONTENT ===" in system_content:
                start_marker = "=== PUBLICATION CONTENT ==="
                end_marker = "=== END PUBLICATION CONTENT ==="
                start_idx = system_content.find(start_marker)
                end_idx = system_content.find(end_marker)
                if start_idx != -1 and end_idx != -1:
                    # Keep everything before and after the publication content
                    before_pub = system_content[:start_idx]
                    after_pub = system_content[end_idx + len(end_marker):]
                    system_content = before_pub + "[PUBLICATION CONTENT OMITTED FOR READABILITY]" + after_pub
            content += f"SYSTEM: {system_content}\n\n"
        elif isinstance(msg, HumanMessage):
            user_question_count += 1
            # Add separator before user messages (except if it's the first message)
            if i > 0:
                content += "=" * 80 + "\n"
            content += f"USER Q{user_question_count}: {msg.content}\n\n"
        elif isinstance(msg, AIMessage):
            content += f"ASSISTANT: {msg.content}\n\n"
    return content


def apply_stuffing_strategy(conversation: list, system_prompt: str) -> list:
    """Strategy 1: Keep all messages."""
    return [SystemMessage(content=system_prompt)] + conversation


def apply_trimming_strategy(conversation: list, system_prompt: str, window_size: int = 8) -> list:
    """Strategy 2: Keep only recent N messages."""
    system_msg = [SystemMessage(content=system_prompt)]
    if len(conversation) <= window_size:
        return system_msg + conversation
    else:
        return system_msg + conversation[-window_size:]


def apply_summarization_strategy(conversation: list, system_prompt: str, llm, max_tokens: int = 1000) -> list:
    """Strategy 3: Summarize old messages, keep recent ones."""
    system_msg = [SystemMessage(content=system_prompt)]
    
    # If conversation is short, no need to summarize
    current_tokens = count_tokens(messages_to_string(system_msg + conversation))
    if current_tokens <= max_tokens:
        return system_msg + conversation
    
    # Keep last 6 messages and summarize the rest
    recent_messages = conversation[-6:] if len(conversation) > 6 else conversation
    older_messages = conversation[:-6] if len(conversation) > 6 else []
    
    if not older_messages:
        return system_msg + conversation
    
    # Create summary
    try:
        older_text = ""
        for msg in older_messages:
            if isinstance(msg, HumanMessage):
                older_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                older_text += f"Assistant: {msg.content}\n"
        
        summary_prompt = f"""Provide a concise summary of this conversation history:

{older_text}

Focus on main topics and key information. Keep under 200 words."""
        
        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary_message = SystemMessage(content=f"Summary of earlier conversation: {summary_response.content}")
        
        return system_msg + [summary_message] + recent_messages
        
    except Exception as e:
        print(f"  ⚠️ Summarization failed, using trimming: {e}")
        return apply_trimming_strategy(conversation, system_prompt, 8)


def run_memory_strategy_conversation(
    publication_content: str, 
    model_name: str, 
    system_prompt_config_name: str, 
    strategy_name: str, 
    user_questions: list,
    app_config: dict
) -> dict:
    """Run conversation for a single memory strategy."""
    
    # Load system prompt config
    prompt_configs = load_yaml_config(PROMPT_CONFIG_FPATH)
    system_prompt_config = prompt_configs.get(system_prompt_config_name)

    if not system_prompt_config:
        raise ValueError(f"System prompt config '{system_prompt_config_name}' not found")

    # Build system prompt
    system_prompt = build_system_prompt_from_config(system_prompt_config, publication_content)

    # Initialize LLM
    llm = ChatGroq(
        model=model_name,
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    # Get memory config
    memory_config = app_config.get("memory_strategies", {})
    window_size = memory_config.get("trimming_window_size", 8)
    max_tokens = memory_config.get("summarization_max_tokens", 1000)

    print(f"\n🔧 Running Strategy: {strategy_name.upper()} | Questions: {len(user_questions)}")
    
    # Track conversation history (without system prompt)
    conversation_history = []
    qa_pairs = []
    token_progression = []

    # Process each question
    for idx, user_input in enumerate(user_questions, 1):
        print(f"  Processing question {idx}/{len(user_questions)}: {user_input[:50]}...")
        
        # Add user message to history
        conversation_history.append(HumanMessage(content=user_input))
        
        # Apply memory strategy to build current prompt
        if strategy_name == "stuffing":
            current_messages = apply_stuffing_strategy(conversation_history[:-1], system_prompt)
        elif strategy_name == "trimming":
            current_messages = apply_trimming_strategy(conversation_history[:-1], system_prompt, window_size)
        elif strategy_name == "summarization":
            current_messages = apply_summarization_strategy(conversation_history[:-1], system_prompt, llm, max_tokens)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Add current question
        current_messages.append(HumanMessage(content=user_input))
        
        # Count tokens before LLM call
        prompt_tokens = count_tokens(messages_to_string(current_messages))
        
        try:
            # Get response
            response = llm.invoke(current_messages)
            response_tokens = count_tokens(response.content)
            
            # Add AI response to history
            conversation_history.append(AIMessage(content=response.content))
            
            # Track this Q&A
            qa_pairs.append({
                'question': user_input,
                'response': response.content
            })
            
            # Track token usage
            token_progression.append({
                'question_num': idx,
                'prompt_tokens': prompt_tokens,
                'response_tokens': response_tokens,
                'total_tokens': prompt_tokens + response_tokens
            })
            
            if idx % 5 == 0:
                print(f"    ✓ Completed {idx} questions, current prompt: {prompt_tokens:,} tokens")
                
        except Exception as e:
            print(f"    ❌ Error at question {idx}: {e}")
            break

    # Generate final prompt for last question
    if user_questions:
        if strategy_name == "stuffing":
            final_messages = apply_stuffing_strategy(conversation_history[:-1], system_prompt)
        elif strategy_name == "trimming":
            final_messages = apply_trimming_strategy(conversation_history[:-1], system_prompt, window_size)
        elif strategy_name == "summarization":
            final_messages = apply_summarization_strategy(conversation_history[:-1], system_prompt, llm, max_tokens)
        
        final_messages.append(HumanMessage(content=user_questions[-1]))
        final_prompt = messages_to_string(final_messages, include_publication=False)  # Exclude publication for readability
        final_response = conversation_history[-1].content if conversation_history else "No response"
    else:
        final_prompt = ""
        final_response = ""

    # Save strategy-specific files
    save_strategy_results(strategy_name, qa_pairs, final_prompt, final_response, token_progression, user_questions)

    # Calculate totals
    total_prompt_tokens = sum(t['prompt_tokens'] for t in token_progression)
    total_response_tokens = sum(t['response_tokens'] for t in token_progression)
    total_tokens = total_prompt_tokens + total_response_tokens

    return {
        "strategy": strategy_name,
        "total_prompt_tokens": total_prompt_tokens,
        "total_response_tokens": total_response_tokens,
        "total_tokens": total_tokens,
        "questions_processed": len(token_progression),
        "token_progression": token_progression
    }


def save_strategy_results(strategy_name: str, qa_pairs: list, final_prompt: str, final_response: str, token_progression: list, questions: list):
    """Save detailed results for a single strategy."""
    
    content = []
    content.append(f"# {strategy_name.upper()} STRATEGY RESULTS")
    content.append("=" * 60)
    content.append("")
    
    # Strategy description
    descriptions = {
        "stuffing": "Keeps ALL previous messages in conversation history.",
        "trimming": "Keeps only the most recent N messages in conversation history.",
        "summarization": "Summarizes older messages and keeps recent messages for context."
    }
    content.append("## Strategy Description")
    content.append(descriptions.get(strategy_name, "Unknown strategy"))
    content.append("")
    
    # Token progression
    content.append("## Token Usage Progression")
    content.append("| Question | Prompt Tokens | Response Tokens | Total |")
    content.append("|----------|---------------|-----------------|-------|")
    for token_data in token_progression:
        content.append(f"| {token_data['question_num']} | {token_data['prompt_tokens']:,} | {token_data['response_tokens']:,} | {token_data['total_tokens']:,} |")
    content.append("")
    
    # Final prompt
    if questions:
        content.append("## Complete Final Prompt for Last Question")
        content.append(f"**Last Question:** '{questions[-1]}'")
        content.append("")
        content.append("```")
        content.append(final_prompt)
        content.append("```")
        content.append("")
        content.append("**Final Response:**")
        content.append("```")
        content.append(final_response)
        content.append("```")
        content.append("")
    
    # All Q&A pairs
    content.append("## All Q&A Pairs")
    content.append("")
    for i, qa in enumerate(qa_pairs, 1):
        content.append(f"### Question {i}")
        content.append(f"**User:** {qa['question']}")
        content.append("")
        content.append(f"**Assistant:** {qa['response']}")
        content.append("")
        content.append("-" * 40)
        content.append("")
    
    # Save file
    filename = f"lesson3a_strategy_{strategy_name}_results.md"
    save_text_to_file(
        "\n".join(content),
        os.path.join(OUTPUTS_DIR, filename),
        header=f"Lesson 3A: {strategy_name.title()} Strategy Results"
    )
    print(f"    ✓ Results saved to {filename}")


def save_comparison_stats(all_stats: list):
    """Save overall comparison statistics."""
    
    content = []
    content.append("# Memory Strategy Comparison - Overall Statistics")
    content.append("=" * 60)
    content.append("")
    
    # Summary table
    content.append("## Token Usage Summary")
    content.append("")
    content.append("| Strategy | Prompt Tokens | Response Tokens | Total Tokens | Questions |")
    content.append("|----------|---------------|-----------------|--------------|-----------|")
    
    for stats in all_stats:
        content.append(f"| {stats['strategy'].title()} | {stats['total_prompt_tokens']:,} | {stats['total_response_tokens']:,} | {stats['total_tokens']:,} | {stats['questions_processed']} |")
    
    content.append("")
    
    # Efficiency comparison
    if len(all_stats) > 1:
        baseline = next((s for s in all_stats if s['strategy'] == 'stuffing'), all_stats[0])
        content.append("## Efficiency Comparison")
        content.append("")
        for stats in all_stats:
            if stats['strategy'] != baseline['strategy']:
                savings = ((baseline['total_tokens'] - stats['total_tokens']) / baseline['total_tokens']) * 100
                content.append(f"- **{stats['strategy'].title()}** vs {baseline['strategy'].title()}: {savings:.1f}% token savings")
        content.append("")
    
    # Analysis
    content.append("## Analysis")
    content.append("")
    content.append("### When to Use Each Strategy")
    content.append("- **Stuffing**: Short conversations where complete context is crucial")
    content.append("- **Trimming**: When only recent context matters and costs need control")
    content.append("- **Summarization**: Balance between context preservation and efficiency")
    
    # Save file
    save_text_to_file(
        "\n".join(content),
        os.path.join(OUTPUTS_DIR, "lesson3a_memory_comparison_stats.md"),
        header="Lesson 3A: Memory Strategy Comparison Statistics"
    )
    print("✓ Comparison statistics saved to lesson3a_memory_comparison_stats.md")


def run_single_strategy():
    """Run a single memory strategy."""
    load_env()
    publication_content = load_publication(publication_external_id='yzN0OCQT7hUS')
    app_config = load_yaml_config(APP_CONFIG_FPATH)
    model_name = app_config.get("llm", "llama-3.1-8b-instant")

    # Let user pick a strategy
    strategies = ["stuffing", "trimming", "summarization"]
    print("\nAvailable strategies:")
    for idx, s in enumerate(strategies, 1):
        print(f"{idx}. {s}")

    choice = input("\nSelect strategy (1-3, default=1): ").strip()
    strategy_map = {"1": "stuffing", "2": "trimming", "3": "summarization"}
    strategy = strategy_map.get(choice, "stuffing")

    # Load questions
    questions_config = load_yaml_config(os.path.join(DATA_DIR, "yzN0OCQT7hUS-sample-questions.yaml"))
    user_questions = questions_config.get("questions", [])
    
    # Ask how many questions
    num_questions = input(f"How many questions to process? (max {len(user_questions)}, default=10): ").strip()
    try:
        num_questions = int(num_questions) if num_questions else 10
        num_questions = min(num_questions, len(user_questions))
    except ValueError:
        num_questions = 10
    
    selected_questions = user_questions[:num_questions]

    # Run strategy
    stats = run_memory_strategy_conversation(
        publication_content=publication_content,
        model_name=model_name,
        system_prompt_config_name="ai_assistant_system_prompt_advanced",
        strategy_name=strategy,
        user_questions=selected_questions,
        app_config=app_config
    )

    print("\n🎯 Final Stats:")
    print(f"Strategy: {stats['strategy']}")
    print(f"Questions processed: {stats['questions_processed']}")
    print(f"Total prompt tokens: {stats['total_prompt_tokens']:,}")
    print(f"Total response tokens: {stats['total_response_tokens']:,}")
    print(f"Total tokens: {stats['total_tokens']:,}")


def run_comparison():
    """Run comparison of all memory strategies."""
    load_env()
    publication_content = load_publication(publication_external_id='yzN0OCQT7hUS')
    app_config = load_yaml_config(APP_CONFIG_FPATH)
    model_name = app_config.get("llm", "llama-3.1-8b-instant")

    # Load questions
    questions_config = load_yaml_config(os.path.join(DATA_DIR, "yzN0OCQT7hUS-sample-questions.yaml"))
    user_questions = questions_config.get("questions", [])
    
    # Ask how many questions
    num_questions = input(f"How many questions to process? (max {len(user_questions)}, default=10): ").strip()
    try:
        num_questions = int(num_questions) if num_questions else 10
        num_questions = min(num_questions, len(user_questions))
    except ValueError:
        num_questions = 10
    
    selected_questions = user_questions[:num_questions]

    strategies = ["stuffing", "trimming", "summarization"]
    all_stats = []

    print(f"\n🏁 Running comparison with {len(selected_questions)} questions...")

    for strategy in strategies:
        stats = run_memory_strategy_conversation(
            publication_content=publication_content,
            model_name=model_name,
            system_prompt_config_name="ai_assistant_system_prompt_advanced",
            strategy_name=strategy,
            user_questions=selected_questions,
            app_config=app_config
        )
        all_stats.append(stats)

    # Save comparison report
    save_comparison_stats(all_stats)
    
    # Print summary
    print("\n📊 COMPARISON RESULTS:")
    print("-" * 60)
    for stats in all_stats:
        print(f"{stats['strategy'].title():15} | {stats['total_tokens']:,} total tokens")


def main():
    """Main entry point."""
    print("=" * 80)
    print("LESSON 3A: MEMORY STRATEGY DEMONSTRATION")
    print("=" * 80)
    
    print("Choose mode:")
    print("1. Run a single strategy")
    print("2. Run comparison of all strategies")
    
    choice = input("\nChoose mode (1 or 2, default=2): ").strip()

    if choice == "1":
        run_single_strategy()
    else:
        run_comparison()

    print("\n🎉 COMPLETE! Check the outputs/ directory for detailed results.")
    print("=" * 80)


if __name__ == "__main__":
    main()