import os
import pytest

@pytest.fixture
def force_mock_model(monkeypatch):
    real_isfile = os.path.isfile

    def make_fake_file(path: str) -> bool:
        # Pretend every .onnx file is missing → forces mock
        if path.lower().endswith(".onnx"):
            return False
        return real_isfile(path)

    monkeypatch.setattr(os.path, "isfile", make_fake_file)
