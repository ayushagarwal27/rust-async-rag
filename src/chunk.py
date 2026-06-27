import tiktoken
import re

ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")

def chunk_text(text:str, source_file:str, doc_type:str,  chunk_size:int = 500, overlap:int = 50) -> list[dict]:
    """
    Split a markdown document into overlapping chunks, treating code blocks
    as atomic units that are never split mid-way.
 
    Args:
        text:        raw markdown content of the document
        source_file: path to the source file, stored as metadata per chunk
        doc_type:    corpus category (e.g. "tokio_tutorial", "tracing_docs")
        chunk_size:  maximum tokens per prose chunk (code blocks may exceed this)
        overlap:     number of tokens carried over from the previous chunk
                     to preserve context at chunk boundaries
 
    Returns:
        list of dicts, each with keys: text, source_file, doc_type, chunk_index
    """

    # step 1 : split the document into alternating prose and code segments
    # this must happen before any token-based splitting so we never cut
    # a ```rust ... ``` block in half
    segments = split_by_code_blocks(text)

    chunks = []
    current_chunk = "" # accumulates prose tokens until chunk_size is hit

    for segment in segments:
        code = segment["text"]

        if segment["type"] == 'code':
            # tiny code block (< 50 tokens), not worth isolating as its own
            # chunk; merge it into the surrounding prose instead
            if  count_tokens(code) < 50:
                current_chunk += "\n\n" + code
                continue

            # normal code block, save any accumulated prose first, then
            # store the code block as its own standalone chunk
            if  current_chunk.strip():
                chunks.append({
                    "text":current_chunk.strip(), 
                    "source_file": source_file, 
                    "doc_type":doc_type, 
                    "chunk_index":len(chunks)
                })
                # carry the overlap tail into the next chunk for context continuity
                current_chunk = get_last_n_tokens(current_chunk, overlap)

            # code block becomes its own chunk regardless of size
            chunks.append({
                "text":code.strip(),
                "source_file":source_file,
                "doc_type":doc_type,
                "chunk_index":len(chunks)
            })

            # reset, code block breaks the flow
            current_chunk = ""
        else:
            # prose segment — split on paragraph boundaries (\n\n) and
            # accumulate paragraphs until we hit chunk_size
            paragraphs = segment["text"].split("\n\n")

            for paragraph in paragraphs:

                if not paragraph.strip():
                    continue

                # would adding this paragraph push us over the limit?
                if count_tokens(current_chunk + paragraph) > chunk_size:
                    # save the current accumulation before it gets too large
                    if current_chunk.strip():
                        chunks.append({
                                "text":current_chunk.strip(), 
                                "source_file": source_file, 
                                "doc_type":doc_type, 
                                "chunk_index":len(chunks)
                            })
                    # slide the window: carry the tail of this chunk forward
                    # so the next chunk has context from the previous one
                    current_chunk = get_last_n_tokens(current_chunk, overlap)
                    
                     # edge case: a single paragraph is already larger than chunk_size
                    # (e.g. a long prose wall with no \n\n breaks) : save it as-is
                    # since we can't split it further without losing coherence,
                    # then use continue to skip the normal += below to avoid
                    # appending it a second time
                    if count_tokens(paragraph) > chunk_size:
                        chunks.append({
                            "text": paragraph.strip(),
                            "source_file": source_file,
                            "doc_type": doc_type,
                            "chunk_index": len(chunks)
                        })
                        current_chunk = get_last_n_tokens(paragraph, overlap)
                        continue

                # accumulate this paragraph into the current chunk
                current_chunk += "\n\n" + paragraph

    # save whatever prose remains after the loop ends
    if current_chunk.strip():
        chunks.append({
            "text":current_chunk.strip(), 
            "source_file": source_file, 
            "doc_type":doc_type, 
            "chunk_index":len(chunks)
        })

    return chunks


def split_by_code_blocks(text:str)-> list[dict]:
    """
    Split markdown text into alternating prose and code segments.
 
    Uses re.split WITHOUT a capturing group, so the code fence delimiters
    are consumed and not returned as separate items. Each part is then
    classified by checking if it starts with a backtick fence.
 
    Note: this means code blocks lose their opening fence in the split,
    which is intentional for chunking purposes (the content is what matters,
    not the fence markers themselves).
 
    Returns:
        list of dicts with keys: type ("prose" | "code"), text
    """
    parts = re.split(r"```[\s\S]*?```",text)

    segments = []
    for part in parts:
        if part.startswith("```"):
            segments.append({"type":"code", "text":part})
        else:
            # strip leading/trailing whitespace from prose sections
            # (avoids empty segments from whitespace between code blocks)
            part.strip()
            segments.append({"type":"prose", "text":part})
    return segments


def count_tokens(text:str)-> int:
    """
    Count the number of tokens in a string using the gpt-4o-mini tokenizer.
 
    Token counts are used (not character counts) because LLM context windows
    are measured in tokens, not characters, a character-based split would
    produce inconsistent chunk sizes across different content types.
    """
    num_tokens = len(ENCODING.encode(text))
    return num_tokens

def get_last_n_tokens(text:str, overlap:int) -> str:
    """
    Return the last `overlap` tokens of `text` as a decoded string.
 
    This is used to seed the next chunk with the tail of the previous one,
    preserving context across chunk boundaries. Using token-level slicing
    (not character-level) ensures the overlap is exactly `overlap` tokens,
    not approximately that many characters.
    """
    tokens = ENCODING.encode(text)
    last_tokens = tokens[-overlap:]    
    return ENCODING.decode(last_tokens)  


if __name__ == "__main__":
    from pathlib import Path

    source_file = "knowledge_base/async_book/02_execution/02_future.md"
    text = Path(source_file).read_text()
    chunks = chunk_text( text=text, source_file=source_file, doc_type="tokio_tutorial")

    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n---- chunk {i} ---")
        print(chunk["text"])
        print(count_tokens(chunk["text"]))

    # should be empty or contain only code blocks (which can't be split further)
    oversized = [i for i, c in enumerate(chunks) if count_tokens(c["text"]) > 500]
    print(f"\nOversized chunks: {oversized}")


