import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from vectordb import chunk_text, _load_gemini_client, GeminiEmbeddingFunction


def test_chunk_text_basic():
    text = "This is a simple sentence with some more words for chunking test."
    chunks = chunk_text(text, chunk_size=5, overlap=2)

    # Ensure correct chunk size and overlap
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk.split()) <= 5 for chunk in chunks)


@patch("vectordb.genai.Client")
@patch("vectordb.os.getenv", return_value="fake_api_key")
def test_load_gemini_client(mock_getenv, mock_client):
    client_instance = MagicMock()
    mock_client.return_value = client_instance

    client = _load_gemini_client()
    assert client == client_instance
    mock_getenv.assert_called_once_with("GEMINI_API_KEY")


@patch.object(GeminiEmbeddingFunction, "__call__")
def test_embedding_function_call(mock_call):
    mock_call.return_value = [[0.1, 0.2, 0.3]]

    fake_client = MagicMock()
    embed_fn = GeminiEmbeddingFunction(client=fake_client)
    docs = ["Hello world"]
    result = embed_fn(docs)

    assert result == [[0.1, 0.2, 0.3]]
    mock_call.assert_called_once_with(docs)
