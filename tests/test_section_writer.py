import pytest

from src.agents.section_writer import StubSectionWriterAgent, resolve_section_output_path


@pytest.mark.unit
def test_resolve_section_output_path_is_deterministic(temp_project_folder):
    path1, rel1 = resolve_section_output_path(temp_project_folder, section_id="Related Work")
    path2, rel2 = resolve_section_output_path(temp_project_folder, section_id="Related Work")

    assert path1 == path2
    assert rel1 == rel2
    assert rel1.endswith(".tex")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stub_section_writer_writes_deterministic_tex(temp_project_folder):
    agent = StubSectionWriterAgent()

    ctx = {
        "project_folder": str(temp_project_folder),
        "section_id": "related_work",
        "section_title": "Related Work",
    }

    result1 = await agent.execute(ctx)
    assert result1.success is True

    out_rel = result1.structured_data.get("output_relpath")
    assert isinstance(out_rel, str)

    out_path = temp_project_folder / out_rel
    assert out_path.exists()

    content1 = out_path.read_text(encoding="utf-8")

    result2 = await agent.execute(ctx)
    assert result2.success is True

    content2 = out_path.read_text(encoding="utf-8")

    assert content1 == content2
    assert content1 == result1.content
    assert content1.endswith("\n")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stub_section_writer_fails_without_project_folder():
    agent = StubSectionWriterAgent()

    result = await agent.execute({})

    assert result.success is False
    assert result.error == "Missing required input: project_folder"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stub_section_writer_defaults_section_id_to_stub(temp_project_folder):
    agent = StubSectionWriterAgent()

    result = await agent.execute(
        {
            "project_folder": str(temp_project_folder),
            "section_title": "Related Work",
        }
    )

    assert result.success is True
    out_rel = result.structured_data.get("output_relpath")
    assert out_rel == "outputs/sections/stub.tex"
    assert (temp_project_folder / out_rel).exists()

    result2 = await agent.execute(
        {
            "project_folder": str(temp_project_folder),
            "section_id": "   ",
            "section_title": "Related Work",
        }
    )

    assert result2.success is True
    out_rel2 = result2.structured_data.get("output_relpath")
    assert out_rel2 == "outputs/sections/stub.tex"
