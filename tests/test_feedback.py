import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

FEEDBACK_PAYLOAD = {
    "document_id": "test-doc-uuid",
    "corrections": [
        {"element_id": "text_042", "user_final": "entrance"}
    ]
}


@patch("app.main.store_corrections", new_callable=AsyncMock)
def test_feedback_returns_204(mock_store):
    resp = client.post("/feedback", json=FEEDBACK_PAYLOAD)
    assert resp.status_code == 204


@patch("app.main.store_corrections", new_callable=AsyncMock)
def test_feedback_calls_store_with_args(mock_store):
    client.post("/feedback", json=FEEDBACK_PAYLOAD)
    mock_store.assert_called_once_with(
        "test-doc-uuid",
        [{"element_id": "text_042", "user_final": "entrance"}]
    )


def test_feedback_rejects_empty_corrections():
    resp = client.post("/feedback", json={"document_id": "x", "corrections": []})
    assert resp.status_code == 422


def test_feedback_rejects_missing_document_id():
    resp = client.post("/feedback", json={"corrections": [{"element_id": "t0", "user_final": "x"}]})
    assert resp.status_code == 422
