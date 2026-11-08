# RAG adapter

`dragonball_rag_adapter.py` supplies the actual RAG execution used by B2 question
generation. It loads the sibling `rag-error-classification` project lazily and
uses its config/models. It needs that project's corpus, configuration and runtime
dependencies. Importing the adapter does not start inference. For another backend,
pass an importable custom adapter to the dynamic-question CLI.
