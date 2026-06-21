import gradio as gr
from query import query


def chat_fn(message, history):
    answer, sources, _ = query(message)
    
    unique_sources = list(dict.fromkeys(sources))  # dedupe, keep order
    sources_text = "\n".join(f"- `{s}`" for s in unique_sources)
    
    return f"{answer}\n\n**Sources:**\n{sources_text}"


demo = gr.ChatInterface(
    fn=chat_fn,
    title="🦀 Async Rust Debugging Assistant",
    description=(
        "Ask about debugging async Rust — deadlocks, stuck futures, "
        "tokio task scheduling, and tracing instrumentation. "
        "Answers are grounded in the tokio docs, tracing crate, and Rust async book."
    ),
    examples=[
        "How do I debug a deadlock in tokio?",
        "What is the difference between spawn and spawn_blocking?",
        "How does the tracing crate work with async functions?",
        "Why does my tokio::spawn silently die when I drop the JoinHandle?",
    ],
)

if __name__ == "__main__":
    demo.launch()