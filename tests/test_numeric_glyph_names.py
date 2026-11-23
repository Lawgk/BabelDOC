from conftest import build_pdf


def _numeric(*codes: int) -> list[str]:
    return [f"C{code}" for code in codes]


# Subset fonts of a PDFlib PLOP 3.0 rewrite of an Advent 3B2 journal article:
# codes renumbered to 1..N, each glyph named after its code in the original
# font, no ToUnicode. Distiller output of 3B2 names glyphs the same way.
TEXT_FONT = _numeric(80, 111, 114, 99, 105, 110, 101, 174, 177, 32)
GREEK_FONT = _numeric(97, 98)  # alpha, beta
SYMBOL_FONT = _numeric(121, 135, 0, 1, 48, 2)  # minus middot times prime dagger plus


def _text(tmp_path, basefont: str, names: list[str], base_encoding=None) -> str:
    from babeldoc.rust_parser_service import parse_pdf_to_il

    encoding = b"<< /Type /Encoding /Differences [1 %s]%s >>" % (
        b" ".join(b"/" + name.encode() for name in names),
        b" /BaseEncoding /" + base_encoding.encode() if base_encoding else b"",
    )
    font = b"<< /Type /Font /Subtype /Type1 /BaseFont /%s /FirstChar 1" % (
        basefont.encode()
    ) + b" /LastChar %d /Widths [%s] /Encoding %s >>" % (
        len(names),
        b" ".join([b"500"] * len(names)),
        encoding,
    )
    codes = bytes(range(1, len(names) + 1))
    shown = codes.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
    path = tmp_path / "doc.pdf"
    path.write_bytes(build_pdf(b"BT /F1 12 Tf 72 700 Td (" + shown + b") Tj ET", font))
    return "".join(c.char_unicode for c in parse_pdf_to_il(path).page[0].pdf_character)


def _undecoded(count: int) -> str:
    return "".join(f"(cid:{cid})" for cid in range(1, count + 1))


def test_numeric_names_decode_through_the_base_encoding(tmp_path):
    assert _text(tmp_path, "BHKGPL+AdvP4A213B", TEXT_FONT) == "Porcineﬁ– "


def test_numeric_names_follow_an_explicit_base_encoding(tmp_path):
    text = _text(tmp_path, "ABCDEF+Body", _numeric(80, 174, 177), "WinAnsiEncoding")
    assert text == "P®±"


def test_codes_outside_the_base_encoding_leave_the_font_undecoded(tmp_path):
    text = _text(tmp_path, "BHKGPP+AdvP4C4E74", SYMBOL_FONT)
    assert text == _undecoded(len(SYMBOL_FONT))


def test_tiny_subsets_stay_undecoded(tmp_path):
    # Two glyphs cannot tell Greek from Latin: decoding alpha and beta as
    # "a" and "b" would silently rewrite every IFN-alpha in the article.
    assert _text(tmp_path, "BHKHCA+AdvP7DA6", GREEK_FONT) == _undecoded(2)


def test_pi_fonts_stay_undecoded(tmp_path):
    text = _text(tmp_path, "BHKHGL+AdvPi1", _numeric(109, 56, 35, 49))
    assert text == _undecoded(4)


def test_names_count_up_to_their_first_period(tmp_path):
    # 3B2 text fonts carry an alternate parenthesis /C40._ among their /C<n>.
    names = [*_numeric(70, 111, 114, 32), "C40._", *_numeric(97, 41)]
    assert _text(tmp_path, "HDCOAK+AdvPS6F00", names) == "For (a)"


# "stiffness coefficients" as an Advent 3B2 text font draws it: 128 and 129
# are its ff and ffi ligatures.
LIGATURE_TEXT = _numeric(115, 116, 105, 128, 110, 101, 115, 115, 32)
LIGATURE_TEXT += _numeric(99, 111, 101, 129, 99, 105, 101, 110, 116, 115)


def test_advent_text_fonts_decode_their_ligatures(tmp_path):
    text = _text(tmp_path, "ICJJLH+AdvPSTim", LIGATURE_TEXT)
    assert text == "stiffness coefficients"


def test_a_few_codes_outside_the_base_encoding_stay_undecoded(tmp_path):
    # Not an Advent font: 128 and 129 mean nothing, the other 21 glyphs decode.
    names = [*LIGATURE_TEXT, *_numeric(32, 97, 110, 100)]
    text = _text(tmp_path, "ABCDEF+Body", names)
    assert text == "sti(cid:4)ness coe(cid:13)cients and"


def test_many_codes_outside_the_base_encoding_leave_the_font_undecoded(tmp_path):
    names = [*LIGATURE_TEXT, *_numeric(130, 32, 97, 110, 100)]
    assert _text(tmp_path, "ABCDEF+Body", names) == _undecoded(len(names))


def test_partly_numeric_names_keep_the_existing_decoding(tmp_path):
    text = _text(tmp_path, "ABCDEF+Body", ["A", *_numeric(66, 67)])
    assert text == "A(cid:2)(cid:3)"
