from pathlib import Path

from skillllm_v2.eval.money_table_generator import generate_money_table


def test_money_table_generator(tmp_path: Path) -> None:
    rows = generate_money_table(tmp_path)
    assert len(rows) >= 6
    assert "Regression vs Raw Mistral" in rows[0]
    assert "Artifact path" in rows[0]
    assert (tmp_path / "money_table.csv").exists()
