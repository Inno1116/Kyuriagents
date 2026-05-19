import zipfile
from pathlib import Path

from deepagents.ingestion.parsers import LocalDocumentParser, MCPDocumentParser, ParseRequest


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


def test_local_document_parser_decodes_plain_text(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("first line\nsecond line", encoding="utf-8")
    parser = LocalDocumentParser()

    parsed = parser.parse(
        ParseRequest(
            file_path=path,
            source_uri="upload://tenant/kb/doc/notes.txt",
            filename="notes.txt",
            mime_type="text/plain",
        )
    )

    assert parsed.title == "notes.txt"
    assert parsed.sections[0].text == "first line\nsecond line"
    assert parsed.metadata["encoding"] == "utf-8-sig"


def test_local_document_parser_extracts_docx_paragraphs(tmp_path):
    path = tmp_path / "sample.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            """
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body>
                <w:p><w:r><w:t>First paragraph</w:t></w:r></w:p>
                <w:p><w:r><w:t>Second</w:t></w:r><w:r><w:tab/></w:r><w:r><w:t>paragraph</w:t></w:r></w:p>
              </w:body>
            </w:document>
            """,
        )
        archive.writestr(
            "docProps/core.xml",
            """
            <cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
              xmlns:dc="http://purl.org/dc/elements/1.1/">
              <dc:title>Sample Title</dc:title>
            </cp:coreProperties>
            """,
        )
    parser = LocalDocumentParser()

    parsed = parser.parse(
        ParseRequest(
            file_path=path,
            source_uri="upload://tenant/kb/doc/sample.docx",
            filename="sample.docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    )

    assert parsed.title == "Sample Title"
    assert [section.text for section in parsed.sections] == ["First paragraph", "Second\tparagraph"]
