import random

import pymupdf
import pytest
from conftest import build_pdf

DRAWING_STAGE = "Generate drawing instructions"


def _flat_figure(paths: int) -> bytes:
    # A scatter plot exported one path per marker, each with its own colour
    # and width and no q/Q around it: the graphic-state chain grows with every
    # marker while only the newest colour and width survive.
    rng = random.Random(7)
    ops = [b"BT /F1 12 Tf 72 740 Td (Scatter plot) Tj ET"]
    for _ in range(paths):
        ops.append(
            b"%.3f %.3f %.3f rg %.2f w %.1f %.1f 2 2 re f"
            % (
                rng.random(),
                rng.random(),
                rng.random(),
                0.1 + rng.random(),
                72 + rng.random() * 460,
                100 + rng.random() * 580,
            )
        )
    return build_pdf(b"\n".join(ops))


def _fills(pdf):
    with pymupdf.open(pdf) as doc:
        return [
            (tuple(round(c, 3) for c in d["fill"]), tuple(round(v, 1) for v in d["rect"]))
            for d in doc[0].get_drawings()
        ]


def test_many_separately_coloured_paths_render_in_linear_time(translate_pdf, tmp_path):
    source = _flat_figure(30_000)
    mono, stages = translate_pdf(source)
    # Walking the whole chain per path took ~20 s here; resolved incrementally
    # it is well under a second.
    assert stages[DRAWING_STAGE] < 5
    (tmp_path / "source.pdf").write_bytes(source)
    assert _fills(mono) == _fills(tmp_path / "source.pdf")


EXT_G_STATES = b" /ExtGState << /GS0 << /CA 0.5 /ca 0.5 >> /GS1 << /CA 1 /ca 1 >> >>"


def _nested_figure(paths: int, seed: int) -> bytes:
    # Paths under random nested q/Q, with colour, width, dash, ExtGState and
    # clip changes between them: superseded and surviving graphic-state
    # operators interleave at every depth, and sibling q/Q blocks share the
    # state before them.
    rng = random.Random(seed)
    ops = [b"BT /F1 14 Tf 72 740 Td (Figure with nested graphics state) Tj ET"]
    depth = 0
    for _ in range(paths):
        r = rng.random()
        if r < 0.12 and depth < 6:
            ops.append(b"q")
            depth += 1
        elif r < 0.22 and depth > 0:
            ops.append(b"Q")
            depth -= 1
        for _ in range(rng.randint(0, 3)):
            k = rng.random()
            if k < 0.3:
                ops.append(b"%.3f %.3f %.3f rg" % (rng.random(), rng.random(), rng.random()))
            elif k < 0.5:
                ops.append(b"%.3f %.3f %.3f RG" % (rng.random(), rng.random(), rng.random()))
            elif k < 0.65:
                ops.append(b"%.2f w" % (0.1 + rng.random()))
            elif k < 0.75:
                ops.append(b"[%d %d] 0 d" % (rng.randint(1, 4), rng.randint(1, 4)))
            elif k < 0.97:
                ops.append(b"/GS%d gs" % rng.randint(0, 1))
            else:
                x, y = rng.random() * 150, rng.random() * 250
                ops.append(b"%.1f %.1f 480 560 re W n" % (x, y))
        x, y = 72 + rng.random() * 460, 100 + rng.random() * 580
        ops.append(b"%.1f %.1f 2.5 2.5 re %s" % (x, y, rng.choice([b"f", b"S", b"B"])))
    ops.extend([b"Q"] * depth)
    return build_pdf(b"\n".join(ops), resources=EXT_G_STATES)


def _painted(pdf):
    """Every painted path with its paint state and the clip it is drawn under."""

    def r(values, digits):
        return None if values is None else tuple(round(v, digits) for v in values)

    painted, clips = [], []
    with pymupdf.open(pdf) as doc:
        for item in doc[0].get_drawings(extended=True):
            # An item at nesting level L ends every clip opened at L or deeper.
            while clips and clips[-1][0] >= item["level"]:
                clips.pop()
            if item["type"] == "clip":
                clips.append((item["level"], r(item["scissor"], 1)))
            elif item["type"] != "group":
                painted.append(
                    (
                        item["type"],
                        r(item.get("fill"), 3),
                        r(item.get("color"), 3),
                        round(item.get("width") or 0, 2),
                        item.get("dashes"),
                        round(item.get("fill_opacity") or 1, 2),
                        round(item.get("stroke_opacity") or 1, 2),
                        r(item["rect"], 1),
                        clips[-1][1] if clips else None,
                    )
                )
    return painted


@pytest.mark.parametrize("seed", [1, 2])
def test_nested_graphics_state_reaches_the_output_unchanged(translate_pdf, tmp_path, seed):
    source = _nested_figure(3_000, seed)
    mono, _ = translate_pdf(source)
    (tmp_path / "source.pdf").write_bytes(source)
    expected = _painted(tmp_path / "source.pdf")
    assert len(expected) == 3_000
    assert _painted(mono) == expected
