import io
import json

from project_alexandria.backends import OpenAICompatibleBackend


def test_openai_backend_retries_empty_completion(monkeypatch):
    responses = iter(
        [
            {"choices": [{"finish_reason": "length", "message": {"content": ""}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": "C;"}}]},
        ]
    )

    def fake_urlopen(request, timeout):
        del request, timeout
        return io.BytesIO(json.dumps(next(responses)).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    backend = OpenAICompatibleBackend("judge", retries=2)

    assert backend.generate("system", "prompt") == "C;"
