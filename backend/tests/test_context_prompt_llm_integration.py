from app.llm.llm_provider import LLMProvider
from app.llm.llm_service import LLMService
from app.rag.context_builder import ContextBuilder
from app.rag.prompt_builder import PromptBuilder
from app.schemas.search_result_schema import SearchResultSchema


class FakeLLMProvider(LLMProvider):
    """
    Deterministic LLM provider used for integration testing.

    It does not call any external LLM API.
    """

    def __init__(self):
        self.received_prompt = None

    def generate(self, prompt: str) -> str:
        self.received_prompt = prompt

        return (
            "Machine learning is a branch of artificial "
            "intelligence that learns patterns from data."
        )

    def get_model_name(self) -> str:
        return "fake-test-llm"


def test_context_prompt_llm_integration():
    """
    Verify the integration:

    Retrieved Results
        ↓
    ContextBuilder
        ↓
    RAGContext
        ↓
    PromptBuilder
        ↓
    Prompt
        ↓
    LLMService
        ↓
    FakeLLMProvider
        ↓
    LLM Response
    """

    # ---------------------------------------------------------
    # STEP 1: Create retrieved search results
    # ---------------------------------------------------------

    results = [
        SearchResultSchema(
            point_id="point-001",
            score=0.95,
            payload={
                "document_id": "doc-001",
                "document_name": "machine_learning.txt",
                "chunk_id": 1,
                "text": (
                    "Machine learning is a branch of artificial "
                    "intelligence that learns patterns from data."
                ),
            },
        )
    ]

    # ---------------------------------------------------------
    # STEP 2: Build context using the real ContextBuilder
    # ---------------------------------------------------------

    context_builder = ContextBuilder()

    context = context_builder.build(results)

    assert context.formatted_context

    assert "machine_learning.txt" in context.formatted_context

    assert (
        "Machine learning is a branch of artificial intelligence"
        in context.formatted_context
    )

    # ---------------------------------------------------------
    # STEP 3: Build prompt using the real PromptBuilder
    # ---------------------------------------------------------

    query = "What is machine learning?"

    prompt_builder = PromptBuilder()

    prompt = prompt_builder.build(
        query=query,
        context=context,
    )

    assert prompt

    assert query in prompt

    assert "machine_learning.txt" in prompt

    assert "doc-001" in prompt

    assert "Chunk ID: 1" in prompt

    assert "Relevance Score: 0.9500" in prompt

    assert (
        "Machine learning is a branch of artificial intelligence"
        in prompt
    )

    assert "RETRIEVED CONTEXT" in prompt

    assert "USER QUESTION" in prompt

    assert "ANSWER" in prompt

    # ---------------------------------------------------------
    # STEP 4: Create real LLMService with fake provider
    # ---------------------------------------------------------

    fake_provider = FakeLLMProvider()

    llm_service = LLMService(
        provider=fake_provider,
    )

    # ---------------------------------------------------------
    # STEP 5: Send generated prompt to LLMService
    # ---------------------------------------------------------

    answer = llm_service.generate(prompt)

    # ---------------------------------------------------------
    # STEP 6: Verify LLM response
    # ---------------------------------------------------------

    assert answer

    assert (
        answer
        == "Machine learning is a branch of artificial "
        "intelligence that learns patterns from data."
    )

    # ---------------------------------------------------------
    # STEP 7: Verify the exact prompt reached the provider
    # ---------------------------------------------------------

    assert fake_provider.received_prompt == prompt

    # ---------------------------------------------------------
    # STEP 8: Verify LLM model metadata
    # ---------------------------------------------------------

    assert llm_service.get_model_name() == "fake-test-llm"