"""Validate that the committed notebooks are well-formed and executed."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = ["01_eda.ipynb", "02_modeling.ipynb"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_valid_and_executed(name: str) -> None:
    path = NOTEBOOKS_DIR / name
    assert path.exists(), f"{name} is missing"

    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)  # raises if malformed

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells, "notebook has no code cells"
    assert any(cell.get("outputs") for cell in code_cells), "notebook has no stored outputs"
