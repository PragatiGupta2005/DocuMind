from pathlib import Path

from app.processors.pptx_processor import PPTXProcessor


# Get the backend directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Test PPTX file
TEST_FILE = (
    BASE_DIR
    / "tests"
    / "fixtures"
    / "sample.pptx"
)


def test_pptx_processor():

    processor = PPTXProcessor()

    result = processor.process(
        str(TEST_FILE)
    )

    print(
        result.model_dump_json(
            indent=4
        )
    )

    assert result is not None