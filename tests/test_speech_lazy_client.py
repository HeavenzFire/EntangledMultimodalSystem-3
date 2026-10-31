from unittest.mock import patch

import pytest

from src.core.speech import SpeechRecognizer
from src.utils.errors import SpeechRecognitionError


def test_speech_recognizer_initializes_without_cloud_credentials():
    with patch("src.core.speech.speech.SpeechClient") as client_factory:
        recognizer = SpeechRecognizer()

    client_factory.assert_not_called()
    assert recognizer._client is None


def test_cloud_client_is_created_only_when_requested():
    sentinel = object()
    with patch("src.core.speech.speech.SpeechClient", return_value=sentinel) as factory:
        recognizer = SpeechRecognizer()
        assert recognizer.client is sentinel
        assert recognizer.client is sentinel

    factory.assert_called_once_with()


def test_cloud_client_credential_failure_is_reported_on_demand():
    with patch(
        "src.core.speech.speech.SpeechClient",
        side_effect=RuntimeError("credentials unavailable"),
    ):
        recognizer = SpeechRecognizer()
        with pytest.raises(
            SpeechRecognitionError, match="Cloud speech client initialization failed"
        ):
            _ = recognizer.client
