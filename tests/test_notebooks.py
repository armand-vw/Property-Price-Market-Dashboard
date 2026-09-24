"""Validate that the committed notebooks are well-formed and executed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = ["01_eda.ipynb", "02_modeling.ipynb"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_is_valid_json_and_executed(name: str) -> None:
    path = NOTEBOOKS_DIR / name
    assert path.exists(), f"{name} is missing"

    notebook = json.loads(path.read_text(encoding="utf-8"))  # valid JSON
    assert notebook.get("nbformat") == 4
    assert isinstance(notebook.get("cells"), list)

    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    assert code_cells, "notebook has no code cells"
    assert any(cell.get("outputs") for cell in code_cells), "notebook has no stored outputs"
