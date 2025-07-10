import pytest
import pathlib
import datetime
from unittest.mock import MagicMock, patch
from cache_utils import (
    load_gemini_client,
    download_file,
    upload_file,
    create_explicit_cache,
    generate_from_cache,
    list_caches,
    get_cache_metadata,
    update_cache_ttl,
    update_cache_expiry_time,
    delete_cache,
    list_files,
    get_file_metadata,
    delete_file,
)

# === TEST CLIENT INITIALIZATION ===

@patch("cache_utils.os.getenv", return_value="fake_key")
@patch("cache_utils.genai.Client")
def test_load_gemini_client(mock_client, mock_getenv):
    client = load_gemini_client()
    mock_client.assert_called_once_with(api_key="fake_key")
    assert client is not None

# === TEST FILE DOWNLOAD ===

@patch("cache_utils.requests.get")
def test_download_file(mock_get, tmp_path):
    # Setup mock response
    mock_response = MagicMock()
    mock_response.iter_content = lambda chunk_size: [b"chunk1", b"chunk2"]
    mock_response.raise_for_status = lambda: None
    mock_get.return_value = mock_response

    dest = tmp_path / "file.txt"
    result = download_file("http://example.com/file.txt", dest)

    assert result.exists()
    assert result.name == "file.txt"

# === TEST FILE UPLOAD ===

def test_upload_file_waits_for_processing():
    fake_file = MagicMock()
    fake_file.state.name = "PROCESSING"
    fake_file.name = "mocked_file"
    fake_file_processed = MagicMock()
    fake_file_processed.state.name = "READY"

    mock_client = MagicMock()
    mock_client.files.upload.return_value = fake_file
    mock_client.files.get.return_value = fake_file_processed

    with patch("time.sleep"):
        result = upload_file(mock_client, "mock_path.txt")

    assert result.state.name == "READY"

# === TEST CACHE METHODS ===

def test_create_explicit_cache():
    mock_client = MagicMock()
    mock_client.caches.create.return_value = {"name": "cache123"}

    result = create_explicit_cache(
        client=mock_client,
        model="gemini-2.0",
        contents=["some_file"],
        system_instruction="Test system",
        ttl_seconds=120,
        display_name="mycache"
    )
    assert result["name"] == "cache123"

def test_generate_from_cache():
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = {"text": "response"}

    result = generate_from_cache(
        mock_client,
        model="gemini-2.0",
        cache_name="my_cache",
        prompt="What is this?"
    )
    assert result["text"] == "response"

def test_list_and_get_cache():
    mock_client = MagicMock()
    mock_client.caches.list.return_value = ["cache1"]
    mock_client.caches.get.return_value = {"name": "cache1"}

    assert list_caches(mock_client) == ["cache1"]
    assert get_cache_metadata(mock_client, "cache1") == {"name": "cache1"}

def test_update_cache_ttl_and_expiry():
    mock_client = MagicMock()
    mock_client.caches.update.return_value = {"name": "cache_updated"}

    now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=10)

    ttl_result = update_cache_ttl(mock_client, "my_cache", 300)
    expiry_result = update_cache_expiry_time(mock_client, "my_cache", now)

    assert ttl_result["name"] == "cache_updated"
    assert expiry_result["name"] == "cache_updated"

def test_delete_cache():
    mock_client = MagicMock()
    delete_cache(mock_client, "my_cache")
    mock_client.caches.delete.assert_called_once_with(name="my_cache")

# === FILE LISTING AND DELETION ===

def test_list_get_delete_files():
    mock_client = MagicMock()
    mock_client.files.list.return_value = ["file1"]
    mock_client.files.get.return_value = {"name": "file1"}

    assert list_files(mock_client) == ["file1"]
    assert get_file_metadata(mock_client, "file1") == {"name": "file1"}

    delete_file(mock_client, "file1")
    mock_client.files.delete.assert_called_once_with(name="file1")
