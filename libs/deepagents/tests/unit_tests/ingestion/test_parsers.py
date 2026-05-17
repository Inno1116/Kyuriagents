from pathlib import Path

from deepagents.ingestion.parsers import MCPDocumentParser, ParseRequest


class FakeParserTool:
    __name__ = "parse_document"

    def invoke(self, input_data):
        assert input_data["filename"] == "sample.pdf"
        return {
            "title": "Sample",
            "language": "en",
            "sections": [
                {"text": "First page text", "page_start": 1, "page_end": 1},
                {"text": "Second page text", "page_start": 2, "page_end": 2},
            ],
        }


def test_mcp_document_parser_normalizes_structured_tool_result():
    parser = MCPDocumentParser(tools=[FakeParserTool()])

    parsed = parser.parse(
        ParseRequest(
            file_path=Path("sample.pdf"),
            source_uri="upload://tenant/kb/doc/sample.pdf",
            filename="sample.pdf",
            mime_type="application/pdf",
        )
    )

    assert parsed.title == "Sample"
    assert parsed.language == "en"
    assert [section.text for section in parsed.sections] == ["First page text", "Second page text"]
    assert parsed.sections[0].page_start == 1
