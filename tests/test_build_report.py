import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "build_report", ROOT / "scripts" / "build_report.py"
)
build_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_report)


def test_h3_style_is_bold_and_distinct_from_body():
    styles = build_report.build_styles()
    assert "h3" in styles
    assert styles["h3"].fontName.endswith("-Bold")
    assert styles["h3"].fontSize > styles["body"].fontSize


def test_eight_column_table_uses_the_full_text_width():
    page_width = 16.6 * build_report.cm
    widths = build_report.table_widths(8, page_width)
    assert len(widths) == 8
    assert sum(widths) == pytest.approx(page_width)
