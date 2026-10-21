from babeldoc.format.pdf.document_il.il_version_1 import Box
from babeldoc.format.pdf.document_il.il_version_1 import PdfCharacter
from babeldoc.format.pdf.document_il.il_version_1 import PdfStyle
from babeldoc.format.pdf.document_il.utils.layout_helper import floating_accent_bases
from babeldoc.format.pdf.document_il.utils.layout_helper import get_char_unicode_string


def _char(text: str, x: float, x2: float, *, y: float = 604.8, font: str = "F4"):
    return PdfCharacter(
        pdf_style=PdfStyle(font_id=font, font_size=9),
        box=Box(x=x, y=y, x2=x2, y2=y + 8.97),
        char_unicode=text,
    )


# "Mittelhäusern" as an Advent 3B2 journal draws it: the bare letter, then a
# spacing dieresis moved back over it.
def _mittelhaeusern(accent_font: str = "F4"):
    return [
        _char("h", 204.16, 208.64),
        _char("a", 208.64, 213.13),
        _char("¨", 209.42, 212.41, y=604.97, font=accent_font),
        _char("u", 213.16, 217.65),
        _char("s", 217.65, 221.1),
    ]


def test_accent_after_its_letter_composes():
    assert get_char_unicode_string(_mittelhaeusern()) == "häus"


def test_accent_before_its_letter_composes():
    # TeX's \accent draws the accent first.
    chars = [
        _char("T", 131.98, 137.49),
        _char("ˆ", 138.56, 141.54, y=607.0),
        _char("A", 136.74, 143.18),
        _char("C", 142.92, 148.89),
    ]
    assert get_char_unicode_string(chars) == "TÂC"


# "TÂC" as drawn in the sample: the circumflex ends 1.38pt before the C.
def _tac():
    return [
        _char("T", 131.98, 137.49),
        _char("A", 136.74, 143.18),
        _char("ˆ", 138.56, 141.54, y=607.0),
        _char("C", 142.92, 148.89),
    ]


def test_word_gap_is_measured_from_the_letter_not_its_accent():
    # Word gaps of 1.2 and 1.3pt make 1.3pt the space threshold, which the
    # 1.38pt after the accent would cross.
    chars = [
        *_tac(),
        _char(" ", 148.89, 151.0),
        _char("x", 170.0, 174.0),
        _char("y", 175.2, 179.0),
        _char("z", 180.3, 184.0),
    ]
    assert get_char_unicode_string(chars).split(" ")[0] == "TÂC"


def test_spaces_elsewhere_are_unchanged():
    # The gap after the accent still counts towards the space threshold, as
    # it did before accents were composed; without it the threshold rises
    # to 3pt and "e A" becomes "eA".
    chars = [
        *_tac(),
        _char("e", 160.0, 164.0),
        _char("A", 166.92, 172.0),
        _char("g", 175.0, 179.0),
    ]
    assert get_char_unicode_string(chars) == "TÂC e A g"


def test_free_standing_accents_stay_as_they_are():
    chars = [_char("a", 208.64, 213.13), _char("¨", 213.13, 216.12)]
    assert floating_accent_bases(chars) == {}
    assert get_char_unicode_string(chars) == "a \u0308"


def test_accent_from_another_font_is_not_composed():
    assert floating_accent_bases(_mittelhaeusern(accent_font="F9")) == {}


def test_bases_point_at_the_letter_under_the_accent():
    assert floating_accent_bases(_mittelhaeusern()) == {2: 1}
