from mcp.server.fastmcp import FastMCP
from src.query import query

mcp = FastMCP('async-rust-debugger')


@mcp.tool()
def explain_stack_trace(stack_trace: str) -> str:
    """
    Explain a Rust async stack trace or error message, identifies what's
    blocking, which future is stuck, and suggests fixes grounded in tokio
    docs, the tracing crate, and the Rust async book.

    Use this when the user pastes a raw panic message, JoinError, or
    a tokio-console output snippet.
    """
    question = f"Explain this async Rust error and suggest how to fix it:\n\n{stack_trace}"
    answer, sources, _ = query(question)
    sources_text = "\n".join(f"- {s}" for s in set(sources))
    return f"{answer}\n\nSources:\n{sources_text}"

@mcp.tool()
def search_async_patterns(question: str) -> str:
    """
    Answer a conceptual question about async Rust debugging : deadlocks, task scheduling, spawn vs spawn_blocking, tracing instrumentation, cancellation, or shutdown patterns.

    Use this for general "how do I..." or "why does X happen" questions,
    as opposed to a specific error message or stack trace.
    """
    answer, sources, _ = query(question)
    sources_text = "\n".join(f"- {s}" for s in set(sources))
    return f"{answer}\n\nSources:\n{sources_text}"


@mcp.tool()
def find_tokio_examples(topic: str) -> str:
    """
    Find working code examples for a specific tokio concept : spawning,channels, select!, shutdown, tracing instrumentation, etc.

    Use this when the user wants to see code, not just an explanation.
    """
    question = f"Show a working code example for: {topic}"
    answer, sources, _ = query(question)
    sources_text = "\n".join(f"- {s}" for s in set(sources))
    return f"{answer}\n\nSources:\n{sources_text}"


if __name__ == "__main__":
    mcp.run()