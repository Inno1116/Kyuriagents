from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool

from deepagents.tools import ToolDescriptor, default_tool_registry, tool_description, tool_name


def _sample_tool(query: str) -> str:
    """Search sample data."""
    return query


def test_default_registry_classifies_builtin_tools() -> None:
    registry = default_tool_registry()

    assert registry.descriptor_for("search_knowledge_base").risk == "read_only"
    assert registry.descriptor_for("write_file").requires_confirmation
    assert registry.descriptor_for("execute").risk == "destructive"


def test_registry_rejects_duplicate_descriptor_without_replace() -> None:
    registry = default_tool_registry()
    registry.register(ToolDescriptor(name="custom"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(ToolDescriptor(name="custom"))


def test_tool_name_and_description_support_structured_tools() -> None:
    tool = StructuredTool.from_function(name="sample_search", func=_sample_tool)

    assert tool_name(tool) == "sample_search"
    assert tool_description(tool) == "Search sample data."
