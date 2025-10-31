"""
Test cases for the /analyse endpoint using FastAPI TestClient.
This file is still in progress and more logics and tests will be added to verify additional behaviours.
"""
import io
from fastapi.testclient import TestClient
from PIL import Image

from image_analyser_backend import app

def test_analyse_image_uses_mock(force_mock_model):
    # Arrange
    client = TestClient(app)
    img = Image.new("RGB", (120, 120), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    files = {"input_img": ("mock.jpg", buf, "image/jpeg")}
    params = {"is_return_visualisation": "false", "min_confidence": "0.25"}

    # Act
    res = client.post("/analyse", files=files, params=params)

    # Assert
    assert res.status_code == 200, res.text  # Analyse Service returns AnalysisResult on success.
    data = res.json()
    assert "objects" in data and isinstance(data["objects"], list)
    assert len(data["objects"]) >= 1  # mock returns at least one detection.
