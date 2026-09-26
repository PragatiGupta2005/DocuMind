import importlib

import app.core.settings as settings


def test_rag_default_top_k_from_environment(monkeypatch):

    monkeypatch.setenv(
        "RAG_DEFAULT_TOP_K",
        "8",
    )

    reloaded_settings = importlib.reload(settings)

    assert reloaded_settings.RAG_DEFAULT_TOP_K == 8


def test_rag_max_top_k_from_environment(monkeypatch):

    monkeypatch.setenv(
        "RAG_MAX_TOP_K",
        "15",
    )

    reloaded_settings = importlib.reload(settings)

    assert reloaded_settings.RAG_MAX_TOP_K == 15


def test_min_relevance_score_from_environment(monkeypatch):

    monkeypatch.setenv(
        "MIN_RELEVANCE_SCORE",
        "0.65",
    )

    reloaded_settings = importlib.reload(settings)

    assert reloaded_settings.MIN_RELEVANCE_SCORE == 0.65


def test_all_rag_settings_from_environment(monkeypatch):

    monkeypatch.setenv(
        "RAG_DEFAULT_TOP_K",
        "7",
    )

    monkeypatch.setenv(
        "RAG_MAX_TOP_K",
        "25",
    )

    monkeypatch.setenv(
        "MIN_RELEVANCE_SCORE",
        "0.60",
    )

    reloaded_settings = importlib.reload(settings)

    assert reloaded_settings.RAG_DEFAULT_TOP_K == 7
    assert reloaded_settings.RAG_MAX_TOP_K == 25
    assert reloaded_settings.MIN_RELEVANCE_SCORE == 0.60


def test_rag_configuration_defaults(monkeypatch):

    monkeypatch.delenv(
        "RAG_DEFAULT_TOP_K",
        raising=False,
    )

    monkeypatch.delenv(
        "RAG_MAX_TOP_K",
        raising=False,
    )

    monkeypatch.delenv(
        "MIN_RELEVANCE_SCORE",
        raising=False,
    )

    reloaded_settings = importlib.reload(settings)

    assert reloaded_settings.RAG_DEFAULT_TOP_K == 5
    assert reloaded_settings.RAG_MAX_TOP_K == 20
    assert reloaded_settings.MIN_RELEVANCE_SCORE == 0.5