from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import wrap


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
LEFT_MARGIN = 50
TOP_MARGIN = 60
BOTTOM_MARGIN = 50
FONT_SIZE = 11
LINE_HEIGHT = 15
MAX_CHARS = 92


def normalize_markdown(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            lines.append("")
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip().upper()
        lines.extend(wrap(line, width=MAX_CHARS) or [""])
    return lines


def paginate(lines: list[str]) -> list[list[str]]:
    lines_per_page = (PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN) // LINE_HEIGHT
    pages: list[list[str]] = []
    for index in range(0, len(lines), lines_per_page):
        pages.append(lines[index : index + lines_per_page])
    return pages


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf_objects(pages: list[list[str]]) -> list[bytes]:
    objects: list[bytes] = []

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(len(pages)))
    objects.append(f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode())

    font_object_number = 3 + len(pages) * 2

    for page_index, page_lines in enumerate(pages):
        page_object_number = 3 + page_index * 2
        content_object_number = page_object_number + 1
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_object_number} 0 R >> >> "
            f"/Contents {content_object_number} 0 R >>"
        ).encode()
        objects.append(page_obj)

        text_lines = ["BT", f"/F1 {FONT_SIZE} Tf"]
        y = PAGE_HEIGHT - TOP_MARGIN
        for line in page_lines:
            escaped = pdf_escape(line)
            text_lines.append(f"1 0 0 1 {LEFT_MARGIN} {y} Tm ({escaped}) Tj")
            y -= LINE_HEIGHT
        text_lines.append("ET")
        stream = "\n".join(text_lines).encode("latin-1", errors="replace")
        content_obj = f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        objects.append(content_obj)

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    return objects


def write_pdf(path: Path, pages: list[list[str]]) -> None:
    objects = build_pdf_objects(pages)
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode())
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())

    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    path.write_bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple PDF from the learning markdown guide.")
    parser.add_argument("--input", type=Path, required=True, help="Input markdown file.")
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path.")
    args = parser.parse_args()

    lines = normalize_markdown(args.input.read_text(encoding="utf-8"))
    pages = paginate(lines)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_pdf(args.output, pages)


if __name__ == "__main__":
    main()
