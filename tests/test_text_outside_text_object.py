from babeldoc.rust_parser_service import parse_pdf_to_il


def _pdf(content: bytes) -> bytes:
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


def _runs(tmp_path, content: bytes) -> dict[str, tuple[float, float]]:
    """Map each word found on the page to the (x, y) of its first character."""
    path = tmp_path / "doc.pdf"
    path.write_bytes(_pdf(content))
    page = parse_pdf_to_il(path).page[0]
    chars = sorted(
        (c for c in page.pdf_character if c.char_unicode.strip()),
        key=lambda c: (-round(c.box.y, 2), c.box.x),
    )
    text = "".join(c.char_unicode for c in chars)
    words = {}
    for word in ("Inside", "Outside", "First", "Second", "Stray", "Proper"):
        index = text.find(word)
        if index >= 0:
            words[word] = (chars[index].box.x, chars[index].box.y)
    return words


def test_text_shown_after_et_is_kept_at_the_stray_tm(tmp_path):
    words = _runs(
        tmp_path,
        b"BT /F1 12 Tf 72 700 Td (Inside) Tj ET\n"
        b"1 0 0 1 72 650 Tm /F1 12 Tf (Outside) Tj\n",
    )
    assert set(words) == {"Inside", "Outside"}
    assert abs(words["Outside"][0] - 72) < 1
    assert words["Outside"][1] < words["Inside"][1] - 40


def test_consecutive_stray_runs_advance(tmp_path):
    words = _runs(
        tmp_path,
        b"1 0 0 1 72 650 Tm /F1 12 Tf (First) Tj (Second) Tj\n",
    )
    assert words["Second"][0] > words["First"][0] + 20
    assert abs(words["Second"][1] - words["First"][1]) < 0.01


def test_real_bt_after_stray_text_resets_as_usual(tmp_path):
    words = _runs(
        tmp_path,
        b"1 0 0 1 300 650 Tm /F1 12 Tf (Stray) Tj\n"
        b"BT /F1 12 Tf 72 600 Td (Proper) Tj ET\n",
    )
    assert abs(words["Proper"][0] - 72) < 1
