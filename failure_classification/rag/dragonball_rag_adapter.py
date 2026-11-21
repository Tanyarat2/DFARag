"""Single-query adapter for the checked-in Dragonball RAG configuration."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
RAG_ROOT = ROOT / "rag-error-classification"
GTE_MODEL_REVISION = "104333d6af6f97649377c2afbde10a7704870c7b"
GTE_REMOTE_CODE_REVISION = "40ced75c3017eb27626c9d4ea981bde21a2662f4"


def check_runtime():
    runtime = os.environ.get("DRAGONBALL_RAG_DEVICE", "auto")
    if runtime not in {"auto", "cpu"}:
        raise ValueError("DRAGONBALL_RAG_DEVICE must be auto or cpu")
    if runtime == "cpu":
        return 0
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "Dragonball inference requires at least one accessible CUDA GPU. "
            "Install or repair the NVIDIA driver, then confirm nvidia-smi works, "
            "or explicitly set DRAGONBALL_RAG_DEVICE=cpu for slower CPU inference."
        )
    return torch.cuda.device_count()


def apply_runtime_placement(config, gpu_count):
    """Keep the configured models while selecting their execution devices."""
    if gpu_count == 0:
        config.embedding.device = "cpu"
        config.rerank.device = "cpu"
        config.generator.device = "cpu"
        config.generator.torch_dtype = "float32"
    elif gpu_count == 1:
        config.embedding.device = "cuda:0"
        config.rerank.device = "cpu"
        config.generator.device = "cuda:0"
        config.generator.torch_dtype = "float16"


class DragonballRAG:
    def __init__(self):
        gpu_count = check_runtime()
        sys.path.insert(0, str(RAG_ROOT))
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        import baseconfig  # Registers the repository's structured Hydra config.
        from base.artifact import DocumentDatasetArtifact
        from clients.local_embedding_client import LocalEmbeddingClient
        from clients.local_numpy_client import LocalNumpyVectorDBClient
        from clients.local_rerank_client import LocalRerankClient
        from clients.local_generate_client import LocalGenerateClient

        with initialize_config_dir(version_base=None, config_dir=str(RAG_ROOT / "conf")):
            configuration = compose(config_name="config-dragonball")
        config = OmegaConf.to_object(configuration.rag)
        config.data.corpus_path = RAG_ROOT / config.data.corpus_path
        config.data.query_path = RAG_ROOT / config.data.query_path
        # Pin the model snapshot and custom Transformers code for repeatable loading.
        config.embedding.revision = GTE_MODEL_REVISION
        config.embedding.code_revision = GTE_REMOTE_CODE_REVISION
        apply_runtime_placement(config, gpu_count)
        self.metadata = OmegaConf.to_container(configuration.rag, resolve=True)
        self.metadata["embedding"]["revision"] = GTE_MODEL_REVISION
        self.metadata["embedding"]["code_revision"] = GTE_REMOTE_CODE_REVISION
        self.metadata["runtime_device_placement"] = {
            "requested_runtime": os.environ.get("DRAGONBALL_RAG_DEVICE", "auto"),
            "embedding": config.embedding.device,
            "reranker": config.rerank.device,
            "generator": config.generator.device,
            "generator_torch_dtype": getattr(config.generator, "torch_dtype", "model_default"),
        }
        self.config = config
        artifact = DocumentDatasetArtifact.from_config(config)
        artifact.load()
        self.chunks = artifact.artifact
        self.embedding = LocalEmbeddingClient(config.embedding)
        self.metadata["runtime_compatibility"] = {
            "embedding_repairs": self.embedding.compatibility_repairs,
        }
        vectors = self.embedding.encode(self.chunks.chunk_texts, config.embedding.batch_size).numpy()
        self.database = LocalNumpyVectorDBClient()
        self.database.add(vectors=vectors, ids=self.chunks.chunk_ids, payloads=self.chunks.chunks)
        self.reranker = LocalRerankClient(config.rerank)
        self.generator = LocalGenerateClient(config.generator)

    def ask(self, question):
        vectors = self.embedding.encode([question], self.config.embedding.batch_size).numpy()
        ids, chunks, scores = self.database.search(vectors, self.config.retrieval.retrieve_topk)
        # Match base.component.Reranker's defaults, rather than the client's
        # different defaults (three passes).
        order = self.reranker.rerank_batch(
            queries=[question], chunks=[[chunk["text"] for chunk in chunks[0]]],
            batch_size=self.config.rerank.batch_size,
            num_passes=1, window_size=1000, step=1000,
        )
        context = chunks[0][order[0, :self.config.rerank.rerank_topk]]
        answers, _ = self.generator.generate_batch(
            query=[question], context_docs=[context], batch_size=self.config.generator.batch_size,
        )
        return {
            "answer": answers[0],
            "retrieved_candidates": chunks[0].tolist(),
            "retrieval_scores": scores[0].tolist(),
            "rerank_order": order[0].tolist(),
            "generation_context": context.tolist(),
            "pipeline_metadata": self.metadata,
            "provenance_note": "New run of checked-in Dragonball configuration; historical model revision equivalence unverified.",
        }


_pipeline = None


def ask_rag(question):
    global _pipeline
    if _pipeline is None:
        _pipeline = DragonballRAG()
    return _pipeline.ask(question)


if __name__ == "__main__":
    check_runtime()
    print("Dragonball CUDA runtime available")
