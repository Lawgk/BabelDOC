from types import SimpleNamespace

import httpx
import openai
import pytest

from babeldoc.translator.translator import OpenAITranslator


def _request():
    return httpx.Request("POST", "http://upstream.invalid/v1/chat/completions")


def _status_error(cls, status):
    return cls(
        "upstream said no",
        response=httpx.Response(status, request=_request()),
        body=None,
    )


class _FakeCompletions:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        message = SimpleNamespace(content=" translated ")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


@pytest.fixture
def translator(monkeypatch):
    for method in (OpenAITranslator.do_translate, OpenAITranslator.do_llm_translate):
        monkeypatch.setattr(method.retry, "sleep", lambda seconds: None)
    t = OpenAITranslator("en", "zh", "model", api_key="key", ignore_cache=True)
    return t


def _use(translator, failures):
    completions = _FakeCompletions(failures)
    translator.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return completions


@pytest.mark.parametrize(
    "failure",
    [
        lambda: openai.APIConnectionError(request=_request()),
        lambda: openai.APITimeoutError(request=_request()),
        lambda: _status_error(openai.InternalServerError, 502),
    ],
    ids=["connection", "timeout", "5xx"],
)
@pytest.mark.parametrize("method", ["do_translate", "do_llm_translate"])
def test_transient_failure_is_retried(translator, failure, method):
    completions = _use(translator, [failure(), failure()])
    result = getattr(translator, method)("hello", rate_limit_params={})
    assert result == "translated"
    assert completions.calls == 3


def test_transient_failures_give_up_well_before_rate_limits(translator):
    completions = _use(
        translator, [openai.APIConnectionError(request=_request()) for _ in range(50)]
    )
    with pytest.raises(Exception):
        translator.do_translate("hello")
    assert completions.calls == 6


def test_rate_limit_keeps_its_long_budget(translator):
    completions = _use(
        translator,
        [_status_error(openai.RateLimitError, 429) for _ in range(20)],
    )
    assert translator.do_translate("hello") == "translated"
    assert completions.calls == 21


def test_bad_request_is_not_retried(translator):
    completions = _use(translator, [_status_error(openai.BadRequestError, 400)])
    with pytest.raises(openai.BadRequestError):
        translator.do_translate("hello")
    assert completions.calls == 1
