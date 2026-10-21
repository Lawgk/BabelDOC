from babeldoc.format.pdf.new_parser.active_direct_font_backend import (
    construct_active_direct_runtime_font,
)
from babeldoc.format.pdf.new_parser.object_model import ActiveLiteral


def _numeric(*codes: int) -> list[str]:
    return [f"C{code}" for code in codes]


# Subset fonts of a PDFlib PLOP 3.0 rewrite of an Advent 3B2 journal article:
# codes renumbered to 1..N, each glyph named after its code in the original
# font, no ToUnicode.
TEXT_FONT = _numeric(80, 111, 114, 99, 105, 110, 101, 174, 177, 32)
GREEK_FONT = _numeric(97, 98)  # alpha, beta
SYMBOL_FONT = _numeric(121, 135, 0, 1, 48, 2)  # minus middot times prime dagger plus


def _font(basefont: str, names: list[str], *, base_encoding: str | None = None):
    encoding: dict[object, object] = {
        "Type": ActiveLiteral("Encoding"),
        "Differences": [1, *(ActiveLiteral(name) for name in names)],
    }
    if base_encoding is not None:
        encoding["BaseEncoding"] = ActiveLiteral(base_encoding)
    return construct_active_direct_runtime_font(
        {
            "Type": ActiveLiteral("Font"),
            "Subtype": ActiveLiteral("Type1"),
            "BaseFont": ActiveLiteral(basefont),
            "FirstChar": 1,
            "Widths": [500] * len(names),
            "Encoding": encoding,
        }
    )


def _text(font, count: int) -> str:
    return "".join(
        font.unicode_text(cid, f"(cid:{cid})") for cid in range(1, count + 1)
    )


def _undecoded(count: int) -> str:
    return "".join(f"(cid:{cid})" for cid in range(1, count + 1))


def test_numeric_names_decode_through_the_base_encoding():
    font = _font("BHKGPL+AdvP4A213B", TEXT_FONT)
    assert _text(font, len(TEXT_FONT)) == "Porcineﬁ– "


def test_numeric_names_follow_an_explicit_base_encoding():
    font = _font("ABCDEF+Body", _numeric(80, 174, 177), base_encoding="WinAnsiEncoding")
    assert _text(font, 3) == "P®±"


def test_codes_outside_the_base_encoding_leave_the_font_undecoded():
    font = _font("BHKGPP+AdvP4C4E74", SYMBOL_FONT)
    assert _text(font, len(SYMBOL_FONT)) == _undecoded(len(SYMBOL_FONT))


def test_tiny_subsets_stay_undecoded():
    # Two glyphs cannot tell Greek from Latin: decoding alpha and beta as
    # "a" and "b" would silently rewrite every IFN-alpha in the article.
    font = _font("BHKHCA+AdvP7DA6", GREEK_FONT)
    assert _text(font, 2) == _undecoded(2)


def test_pi_fonts_stay_undecoded():
    font = _font("BHKHGL+AdvPi1", _numeric(109, 56, 35, 49))
    assert _text(font, 4) == _undecoded(4)


def test_partly_numeric_names_keep_the_existing_decoding():
    font = _font("ABCDEF+Body", ["A", *_numeric(66, 67)])
    assert _text(font, 3) == "A(cid:2)(cid:3)"
