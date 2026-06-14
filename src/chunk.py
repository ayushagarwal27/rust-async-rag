import tiktoken
import re

ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")

def chunk_text(text:str, source_file:str, doc_type:str,  chunk_size:int = 500, overlap:int = 50) -> list[dict]:

    segments = split_by_code_blocks(text)

    chunks = []
    current_chunk = ""

    for segment in segments:
        code = segment["text"]

        if segment["type"] == 'code':
            if  count_tokens(code) < 50:
                current_chunk += "\n\n" + code
                continue

            # code block — never split it
            # if current chunk has content, save it first
            if  current_chunk.strip():
                chunks.append({
                    "text":current_chunk.strip(), 
                    "source_file": source_file, 
                    "doc_type":doc_type, 
                    "chunk_index":len(chunks)
                })
                current_chunk = get_last_n_tokens(current_chunk, overlap)

            # code block becomes its own chunk regardless of size
            chunks.append({
                "text":code.strip(),
                "source_file":source_file,
                "doc_type":doc_type,
                "chunk_index":len(chunks)
            })

            # reset — code block breaks the flow
            current_chunk = ""
        else:
            paragraphs = segment["text"].split("\n\n")

            for paragraph in paragraphs:

                if not paragraph.strip():
                    continue

                # If adding this paragraph exceeds chunk_size tokens, save and slide window
                if count_tokens(current_chunk + paragraph) > chunk_size:
                    if current_chunk.strip():
                        chunks.append({
                                "text":current_chunk.strip(), 
                                "source_file": source_file, 
                                "doc_type":doc_type, 
                                "chunk_index":len(chunks)
                            })
                    current_chunk = get_last_n_tokens(current_chunk, overlap)
                    
                    if count_tokens(paragraph) > chunk_size:
                        chunks.append({
                            "text": paragraph.strip(),
                            "source_file": source_file,
                            "doc_type": doc_type,
                            "chunk_index": len(chunks)
                        })
                        current_chunk = get_last_n_tokens(paragraph, overlap)
                        continue

                current_chunk += "\n\n" + paragraph

    
    if current_chunk.strip():
        chunks.append({
            "text":current_chunk.strip(), 
            "source_file": source_file, 
            "doc_type":doc_type, 
            "chunk_index":len(chunks)
        })

    return chunks


def split_by_code_blocks(text:str)-> list[dict]:
    parts = re.split(r"```[\s\S]*?```",text)

    segments = []
    for part in parts:
        if part.startswith("```"):
            segments.append({"type":"code", "text":part})
        else:
            part.strip()
            segments.append({"type":"prose", "text":part})
    return segments


def count_tokens(text:str)-> int:
    num_tokens = len(ENCODING.encode(text))
    return num_tokens

def get_last_n_tokens(text:str, overlap:int) -> str:
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

    # check for oversized chunks
    oversized = [i for i, c in enumerate(chunks) if count_tokens(c["text"]) > 500]
    print(f"\nOversized chunks: {oversized}")


