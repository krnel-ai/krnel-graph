import httpx
from krnel.graph.dataset_ops import LoadDatasetOp
from krnel.graph.llm_ops import LLMEmbedOp
from krnel.runners.local_runner import LoadLocalParquetDatasetOp, LocalArrowRunner


class LocalOllamaRunner(LocalArrowRunner):
    """
    OllamaLocalRunner is a runner that knows how
    to run LLM operations locally using an Ollama local server.
    """
    def __init__(
        self,
        cache_folder: str,
        ollama_server_url: str = "http://localhost:11434",
    ):
        super().__init__(cache_folder)
        self.ollama_server_url = ollama_server_url


if __name__ == "__main__":

    runner = LocalOllamaRunner(cache_folder='/tmp/cache')

    dataset = runner.from_parquet('/Users/kimmy/krnel/research/data/csvs/krnel_harmful_20250204.parquet')

    embeddings = dataset.col_prompt('prompt').llm_embed(
        model_name='ollama:llama3.2:latest',
        layer_num=-1,
        token_mode='last',
    )

    model = embeddings.train_classifier(
        model_name="my_model",
        labels=dataset.col_categorical('label_column'),
        train_test_split=dataset.col_train_test_split('train_test_split'),
    )


    from rich import print
    print(dataset.model_dump())
    print(dataset)
    print(dataset.col_prompt('prompt').model_dump())
    #print(embeddings.model_dump())
    #print(runner.materialize(embeddings))
