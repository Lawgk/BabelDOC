WORDS = ("Inside", "Outside", "First", "Second", "Stray", "Proper")


def test_text_shown_after_et_is_kept_at_the_stray_tm(page_words):
    words = page_words(
        b"BT /F1 12 Tf 72 700 Td (Inside) Tj ET\n"
        b"1 0 0 1 72 650 Tm /F1 12 Tf (Outside) Tj\n",
        WORDS,
    )
    assert set(words) == {"Inside", "Outside"}
    assert abs(words["Outside"][0] - 72) < 1
    assert words["Outside"][1] < words["Inside"][1] - 40


def test_consecutive_stray_runs_advance(page_words):
    words = page_words(
        b"1 0 0 1 72 650 Tm /F1 12 Tf (First) Tj (Second) Tj\n",
        WORDS,
    )
    assert words["Second"][0] > words["First"][0] + 20
    assert abs(words["Second"][1] - words["First"][1]) < 0.01


def test_real_bt_after_stray_text_resets_as_usual(page_words):
    words = page_words(
        b"1 0 0 1 300 650 Tm /F1 12 Tf (Stray) Tj\n"
        b"BT /F1 12 Tf 72 600 Td (Proper) Tj ET\n",
        WORDS,
    )
    assert abs(words["Proper"][0] - 72) < 1
