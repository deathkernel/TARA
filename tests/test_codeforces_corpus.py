from pathlib import Path

from src.codeforces_corpus import build, record_to_text


def test_record_to_text_includes_problem_editorial_and_tests() -> None:
    text = record_to_text(
        {
            "id": "1/A",
            "title": "Two Sum",
            "description": "Find two values.",
            "input_format": "n",
            "output_format": "indices",
            "examples": [{"input": "2", "output": "1 2"}],
            "editorial": "Use a hash map.",
            "official_tests": [{"input": "2", "output": "Test passed"}],
            "tags": ["implementation"],
        }
    )
    assert "Problem: Two Sum" in text
    assert "Editorial / approach:" in text
    assert "Use a hash map." in text
    assert "Official verification tests:" in text
    assert "Tags: implementation" in text


def test_build_separates_editorial_bearing_records(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(
        '{"id":"2/A","title":"B","description":"Desc B","editorial":"Approach B"}\n'
        '{"id":"1/A","title":"A","description":"Desc A","editorial":""}\n',
        encoding="utf-8",
    )

    manifest = build(source, tmp_path / "out")
    assert manifest["source_records"] == 2
    assert manifest["problem_records"] == 2
    assert manifest["reasoning_records"] == 1

    problem_lines = (tmp_path / "out" / "problem_corpus.jsonl").read_text(encoding="utf-8").splitlines()
    reasoning_lines = (tmp_path / "out" / "reasoning_corpus.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(problem_lines) == 2
    assert len(reasoning_lines) == 1
    assert '"id": "2/A"' in reasoning_lines[0]
