TEXT = b"BT /F1 12 Tf 72 700 Td (Kept) Tj ET\n"
# A 2x1 grey image of the bytes 00 ff, in each ASCII encoding.
IMAGE = b"q BI /W 2 /H 1 /BPC 8 /CS /G "


def test_ascii_hex_image_ends_at_its_marker(page_words):
    # As pdfTeX writes it: EI straight after the end marker, no whitespace.
    words = page_words(IMAGE + b"/F /AHx ID 00ff>EI Q\n" + TEXT, ["Kept"])
    assert "Kept" in words


def test_ascii85_image_ends_at_its_marker(page_words):
    words = page_words(IMAGE + b"/F [/A85] ID !<3~>EI Q\n" + TEXT, ["Kept"])
    assert "Kept" in words


def test_text_between_two_images_is_kept(page_words):
    # The EI search alone skips the first image's ">EI" and ends it at the
    # second image's EI, swallowing the text in between.
    first = IMAGE + b"/F /ASCIIHexDecode ID 00ff>EI Q\n"
    second = IMAGE + b"/F /AHx ID 00ff> EI Q\n"
    words = page_words(first + TEXT + second, ["Kept"])
    assert "Kept" in words


def test_ascii_hex_image_without_marker_falls_back_to_the_ei_search(page_words):
    words = page_words(IMAGE + b"/F /AHx ID 00ff EI Q\n" + TEXT, ["Kept"])
    assert "Kept" in words
