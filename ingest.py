"""Embed the trip-plan text files and save them to a local FAISS index.

Run once (and again whenever a file in data/trips changes):
    uv run python ingest.py
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from embeddings import get_embeddings

DESTINATIONS = {
    "saint_martin": "Saint Martin's Island",
    "khulna": "Khulna and the Sundarbans",
    "sylhet_sunamganj": "Sylhet and Sunamganj",
}


def load_documents() -> list[Document]:
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.name, "destination": DESTINATIONS.get(path.stem, path.stem)},
        )
        for path in sorted(config.TRIPS_DIR.glob("*.txt"))
    ]


def main() -> None:
    docs = load_documents()
    if not docs:
        raise SystemExit(f"No .txt files found in {config.TRIPS_DIR}")
    print(f"Loaded {len(docs)} trip plans: {[d.metadata['source'] for d in docs]}")

    # Split on the "== Section ==" headings first so each chunk stays on one topic
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n== ", "\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(docs)

    # Prefix the destination so a chunk like "== Food ==" still says which trip it is
    for chunk in chunks:
        chunk.page_content = f"[{chunk.metadata['destination']}]\n{chunk.page_content.strip()}"
    print(f"Split into {len(chunks)} chunks")

    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(config.VECTOR_STORE_DIR))
    print(f"Saved FAISS index to {config.VECTOR_STORE_DIR}")


if __name__ == "__main__":
    main()
