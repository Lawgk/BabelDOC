import pytest


def build_pdf(content: bytes) -> bytes:
    """One US-Letter page drawing `content`, with Helvetica (WinAnsi) as /F1."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica"
        b" /Encoding /WinAnsiEncoding >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.7\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


@pytest.fixture
def page_words(tmp_path):
    """Parse a one-page PDF; map each requested word to its first glyph's (x, y)."""
    from babeldoc import _rust_il
    from babeldoc.rust_parser_service import parse_pdf_to_il

    def run(content: bytes, words) -> dict[str, tuple[float, float]]:
        path = tmp_path / "doc.pdf"
        path.write_bytes(build_pdf(content))
        # Rust log records reach Python logging only inside a logging run,
        # which do_translate holds for the whole translation.
        _rust_il.acquire_logging_run()
        try:
            page = parse_pdf_to_il(path).page[0]
        finally:
            try:
                _rust_il.flush_logging_run()
            finally:
                _rust_il.stop_logging_run()
        chars = sorted(
            (c for c in page.pdf_character if c.char_unicode.strip()),
            key=lambda c: (-round(c.box.y, 2), c.box.x),
        )
        text = "".join(c.char_unicode for c in chars)
        found = {}
        for word in words:
            index = text.find(word)
            if index >= 0:
                found[word] = (chars[index].box.x, chars[index].box.y)
        return found

    return run
