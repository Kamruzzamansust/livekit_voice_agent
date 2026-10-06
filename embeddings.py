"""English embeddings via fastembed (ONNX, no PyTorch).

BAAI/bge-small-en-v1.5 is ~67 MB and fast on CPU. It downloads into models/fastembed/
on first use.
"""

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings

import config


def get_embeddings() -> Embeddings:
    return FastEmbedEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        cache_dir=str(config.MODELS_DIR / "fastembed"),
    )
