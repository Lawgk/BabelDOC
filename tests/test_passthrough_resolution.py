import random

import pymupdf
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
