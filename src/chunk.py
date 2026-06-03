import tiktoken

ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")

def chunk_text(text:str, chunk_size:int = 500, overlap:int = 50) -> list[dict]:
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
         # If adding this paragraph exceeds chunk_size tokens, save and slide window
        if count_tokens(current_chunk + paragraph) > chunk_size:
            chunks.append({"text":current_chunk.strip()})

            # Overlap: keep the last N tokens as context for the next chunk
            current_chunk = get_last_n_tokens(current_chunk, overlap)
        current_chunk += "\n\n" + paragraph

    
    if current_chunk.strip():
        chunks.append({"text":current_chunk.strip()})

    return chunks


def count_tokens(text:str)-> int:
    num_tokens = len(ENCODING.encode(text))
    return num_tokens

def get_last_n_tokens(text:str, overlap:int) -> str:
    tokens = ENCODING.encode(text)
    last_tokens = tokens[-overlap:]    
    return ENCODING.decode(last_tokens)  


if __name__ == "__main__":
    from pathlib import Path

    text = Path("knowledge_base/tokio/tutorial/async.md").read_text()
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    print(f"Total chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n---- chunk {i} ---")
        print(chunk["text"])
        print(count_tokens(chunk["text"]))

    oversized = [i for i, c in enumerate(chunks) if count_tokens(c["text"]) > 500]
    print(f"\nOversized chunks: {oversized}")


