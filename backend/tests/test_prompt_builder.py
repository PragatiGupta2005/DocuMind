import pytest

from app.rag.prompt_builder import PromptBuilder
from app.schemas.rag_context_schema import (
    ContextChunk,
    RAGContext,
)


def create_context():
    chunk = ContextChunk(
        document_id="doc-001",
        document_name="test.pdf",
        chunk_id=1,
        text="Machine learning allows systems to learn from data.",
        score=0.95,
    )

    return RAGContext(
        chunks=[chunk],
        formatted_context=(
            "[Source 1]\n"
            "Document: test.pdf\n"
            "Document ID: doc-001\n"
            "Chunk ID: 1\n"
            "Relevance Score: 0.9500\n"
            "Content:\n"
            "Machine learning allows systems to learn from data."
        ),
    )


def test_prompt_builder_includes_query():

    builder = PromptBuilder()

    context = create_context()

    prompt = builder.build(
        query="What is machine learning?",
        context=context,
    )

    assert (
        "What is machine learning?"
        in prompt
    )


def test_prompt_builder_includes_context():

    builder = PromptBuilder()

    context = create_context()

    prompt = builder.build(
        query="What is machine learning?",
        context=context,
    )

    assert (
        "Machine learning allows systems to learn from data."
        in prompt
    )


def test_prompt_builder_includes_source_information():

    builder = PromptBuilder()

    context = create_context()

    prompt = builder.build(
        query="What is machine learning?",
        context=context,
    )

    assert "test.pdf" in prompt
    assert "doc-001" in prompt
    assert "Chunk ID: 1" in prompt


def test_prompt_builder_includes_grounding_instructions():

    builder = PromptBuilder()

    context = create_context()

    prompt = builder.build(
        query="What is machine learning?",
        context=context,
    )

    assert (
        "Do not invent facts"
        in prompt
    )

    assert (
        "provided context"
        in prompt
    )


def test_prompt_builder_handles_empty_context():

    builder = PromptBuilder()

    context = RAGContext(
        chunks=[],
        formatted_context="",
    )

    prompt = builder.build(
        query="What is machine learning?",
        context=context,
    )

    assert (
        "No relevant document context was found."
        in prompt
    )


def test_prompt_builder_rejects_empty_query():

    builder = PromptBuilder()

    context = create_context()

    with pytest.raises(ValueError):

        builder.build(
            query="",
            context=context,
        )


def test_prompt_builder_strips_query_whitespace():

    builder = PromptBuilder()

    context = create_context()

    prompt = builder.build(
        query="   What is machine learning?   ",
        context=context,
    )

    assert (
        "USER QUESTION\n"
        "What is machine learning?"
        in prompt
    )


def test_prompt_builder_has_answer_section():

    builder = PromptBuilder()

    context = create_context()

    prompt = builder.build(
        query="What is machine learning?",
        context=context,
    )

    assert "ANSWER" in prompt