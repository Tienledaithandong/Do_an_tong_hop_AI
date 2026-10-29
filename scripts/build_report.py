"""Render report/report.md to report/report.pdf.

A small Markdown subset is supported on purpose: headings, paragraphs, bullet
and numbered lists, fenced code, pipe tables, images and the inline **bold**,
*italic* and `code` spans. Fonts are chosen for Vietnamese coverage.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "report" / "report.md"
TARGET = ROOT / "report" / "report.pdf"

FONT_DIR = Path("C:/Windows/Fonts")
FAMILY = "Segoe" if (FONT_DIR / "segoeui.ttf").exists() else "DejaVu"


def register_fonts() -> None:
    if FAMILY == "Segoe":
        files = {
            "Segoe": "segoeui.ttf",
            "Segoe-Bold": "segoeuib.ttf",
            "Segoe-Italic": "segoeuii.ttf",
            "Segoe-BoldItalic": "segoeuiz.ttf",
        }
    else:
        import matplotlib

        base = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        files = {
            "DejaVu": str(base / "DejaVuSans.ttf"),
            "DejaVu-Bold": str(base / "DejaVuSans-Bold.ttf"),
            "DejaVu-Italic": str(base / "DejaVuSans-Oblique.ttf"),
            "DejaVu-BoldItalic": str(base / "DejaVuSans-BoldOblique.ttf"),
        }

    for name, filename in files.items():
        path = Path(filename) if Path(filename).is_absolute() else FONT_DIR / filename
        pdfmetrics.registerFont(TTFont(name, str(path)))

    import matplotlib

    mono = (
        Path(matplotlib.__file__).parent
        / "mpl-data"
        / "fonts"
        / "ttf"
        / "DejaVuSansMono.ttf"
    )
    pdfmetrics.registerFont(TTFont("Mono", str(mono)))
    pdfmetrics.registerFontFamily(
        FAMILY,
        normal=FAMILY,
        bold=f"{FAMILY}-Bold",
        italic=f"{FAMILY}-Italic",
        boldItalic=f"{FAMILY}-BoldItalic",
    )


def build_styles() -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle(
            "title", fontName=f"{FAMILY}-Bold", fontSize=17, leading=23,
            spaceAfter=8, alignment=TA_LEFT,
        ),
        "h1": ParagraphStyle(
            "h1", fontName=f"{FAMILY}-Bold", fontSize=14, leading=19,
            spaceBefore=16, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2", fontName=f"{FAMILY}-Bold", fontSize=11.5, leading=16,
            spaceBefore=12, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", fontName=FAMILY, fontSize=10.2, leading=15.2,
            spaceAfter=6, alignment=TA_LEFT,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName=FAMILY, fontSize=10.2, leading=15.0,
            leftIndent=14, bulletIndent=4, spaceAfter=3,
        ),
        "code": ParagraphStyle(
            "code", fontName="Mono", fontSize=8.6, leading=12.4,
            leftIndent=12, backColor=colors.HexColor("#f2f2f2"),
            borderPadding=5, spaceBefore=4, spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "caption", fontName=f"{FAMILY}-Italic", fontSize=8.8, leading=12,
            alignment=TA_CENTER, spaceBefore=4, spaceAfter=10,
            textColor=colors.HexColor("#333333"),
        ),
        "cell": ParagraphStyle("cell", fontName=FAMILY, fontSize=8.4, leading=11),
    }


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font name="Mono" size="9">\1</font>', text)
    return text


def parse_blocks(lines: list[str]):
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        if not line.strip():
            i += 1
            continue

        if line.startswith("```"):
            i += 1
            body = []
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            yield ("code", "\n".join(body))
            continue

        if line.startswith("# "):
            yield ("title", line[2:].strip())
            i += 1
            continue
        if line.startswith("### "):
            yield ("h3", line[4:].strip())
            i += 1
            continue
        if line.startswith("## "):
            yield ("h1", line[3:].strip())
            i += 1
            continue

        if line.strip() in ("---", "***"):
            i += 1
            continue

        if line.startswith("!["):
            match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if match:
                yield ("img", match.group(1), match.group(2))
            i += 1
            continue

        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                row = lines[i].strip().strip("|")
                if not re.fullmatch(r"[\s\-:|]+", row):
                    rows.append([cell.strip() for cell in row.split("|")])
                i += 1
            yield ("table", rows)
            continue

        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:].strip())
                i += 1
                while (
                    i < len(lines)
                    and lines[i].strip()
                    and not re.match(r"^([-|#`!]|\d+\.)", lines[i].strip())
                ):
                    items[-1] += " " + lines[i].strip()
                    i += 1
            yield ("ul", items)
            continue

        if re.match(r"^\d+\.\s", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i]):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i]).strip())
                i += 1
                while (
                    i < len(lines)
                    and lines[i].strip()
                    and not re.match(r"^([-|#`!]|\d+\.)", lines[i].strip())
                ):
                    items[-1] += " " + lines[i].strip()
                    i += 1
            yield ("ol", items)
            continue

        paragraph = [line.strip()]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(r"^([-|#`!]|\d+\.\s|---)", lines[i].strip())
        ):
            paragraph.append(lines[i].strip())
            i += 1
        yield ("p", " ".join(paragraph))


def build() -> Path:
    register_fonts()
    styles = build_styles()
    story: list = []
    page_width = A4[0] - 4.4 * cm

    def make_table(rows: list[list[str]]) -> Table:
        data = [
            [Paragraph(inline(cell), styles["cell"]) for cell in row]
            for row in rows
        ]
        n_cols = len(rows[0])
        if n_cols == 7:
            widths = [2.0 * cm, 2.4 * cm, 2.2 * cm, 2.0 * cm, 1.9 * cm, 3.2 * cm, 1.9 * cm]
        else:
            widths = [page_width / n_cols] * n_cols
        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                    ("FONTNAME", (0, 0), (-1, 0), f"{FAMILY}-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return table

    for block in parse_blocks(SOURCE.read_text(encoding="utf-8").splitlines()):
        kind = block[0]
        if kind == "title":
            story.append(Paragraph(inline(block[1]), styles["title"]))
        elif kind == "h1":
            story.append(Paragraph(inline(block[1]), styles["h1"]))
        elif kind == "h3":
            story.append(Paragraph(inline(block[1]), styles["body"]))
        elif kind == "p":
            story.append(Paragraph(inline(block[1]), styles["body"]))
        elif kind == "code":
            body = block[1].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            body = body.replace("\n", "<br/>").replace(" ", "&nbsp;")
            story.append(Paragraph(body, styles["code"]))
        elif kind == "ul":
            for item in block[1]:
                story.append(
                    Paragraph(inline(item), styles["bullet"], bulletText="•")
                )
            story.append(Spacer(1, 4))
        elif kind == "ol":
            for number, item in enumerate(block[1], start=1):
                story.append(
                    Paragraph(inline(item), styles["bullet"], bulletText=f"{number}.")
                )
            story.append(Spacer(1, 4))
        elif kind == "table":
            story.append(Spacer(1, 3))
            story.append(make_table(block[1]))
            story.append(Spacer(1, 9))
        elif kind == "img":
            alt, rel = block[1], block[2]
            path = ROOT / "report" / rel
            if not path.exists():
                path = ROOT / rel
            with PILImage.open(path) as image:
                width_px, height_px = image.size
            width = page_width
            height = width * height_px / width_px
            max_height = 11.5 * cm
            if height > max_height:
                height = max_height
                width = height * width_px / height_px
            story.append(
                KeepTogether(
                    [
                        Spacer(1, 4),
                        Image(str(path), width=width, height=height),
                        Paragraph(inline(alt), styles["caption"]),
                    ]
                )
            )

    doc = SimpleDocTemplate(
        str(TARGET),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title="Bầy ong ra quyết định trên đồ thị",
        author="Đồ án tổng hợp",
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont(FAMILY, 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawRightString(
            A4[0] - 2.2 * cm, 1.3 * cm, f"trang {document.page}"
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return TARGET


if __name__ == "__main__":
    print("wrote", build())
