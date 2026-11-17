import json
import re

import pytest


HELVETICA = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"


def build_pdf(content: bytes, font: bytes = HELVETICA, extra_fonts=None) -> bytes:
    """One US-Letter page drawing `content`, with `font` (Helvetica) as /F1.

    `extra_fonts` maps further resource names (b"F2") to font dictionaries.
    """
    extra_fonts = extra_fonts or {}
    names = b"".join(
        b" /%s %d 0 R" % (name, 6 + index) for index, name in enumerate(extra_fonts)
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
        b" /Resources << /Font << /F1 4 0 R%s >> >> /Contents 5 0 R >>" % names,
        font,
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        *extra_fonts.values(),
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


def _last_input_array(prompt: str):
    """The last JSON array of {id, input} items in a batch prompt, if any."""
    decoder = json.JSONDecoder()
    found = None
    for match in re.finditer(r"\[", prompt):
        try:
            value, _ = decoder.raw_decode(prompt, match.start())
        except ValueError:
            continue
        if (
            isinstance(value, list)
            and value
            and all(isinstance(i, dict) and {"id", "input"} <= i.keys() for i in value)
        ):
            found = value
    return found


class _EchoTranslator:
    """Translator stand-in that returns its input; no network, no cache."""

    @staticmethod
    def build():
        from babeldoc.translator.translator import BaseTranslator

        class Echo(BaseTranslator):
            name = "echo"

            def __init__(self):
                super().__init__("en", "zh", ignore_cache=True)
                self.model = "echo"
                self.inputs: list[str] = []

            def do_translate(self, text, rate_limit_params: dict = None):
                self.inputs.append(text)
                return text

            def do_llm_translate(self, text, rate_limit_params: dict = None):
                if text is None:  # capability probe
                    return None
                batch = _last_input_array(text)
                if batch is not None:
                    self.inputs.extend(str(i["input"]) for i in batch)
                    return json.dumps(
                        [{"id": i["id"], "output": i["input"]} for i in batch],
                        ensure_ascii=False,
                    )
                # Single-paragraph fallback: the paragraph follows this marker.
                paragraph = text.rsplit("Now translate the following text:\n\n", 1)[-1]
                self.inputs.append(paragraph)
                return paragraph

        return Echo()


@pytest.fixture
def translate_pdf(tmp_path):
    """Run the whole pipeline on raw PDF bytes; return (mono path, stage times).

    The texts sent for translation are left in `translate_pdf.inputs`.
    """
    import asyncio

    from babeldoc.format.pdf import high_level
    from babeldoc.format.pdf.translation_config import TranslationConfig
    from babeldoc.format.pdf.translation_config import WatermarkOutputMode

    high_level.init()

    def run(pdf: bytes, **options):
        source = tmp_path / "in.pdf"
        source.write_bytes(pdf)
        translator = _EchoTranslator.build()
        run.inputs = translator.inputs
        config = TranslationConfig(
            translator,
            str(source),
            "en",
            "zh",
            None,
            output_dir=str(tmp_path / "out"),
            working_dir=str(tmp_path / "work"),
            no_dual=True,
            watermark_output_mode=WatermarkOutputMode.NoWatermark,
            use_rich_pbar=False,
            **options,
        )
        stages: dict[str, float] = {}

        async def drive():
            import time

            opened = {}
            async for event in high_level.async_translate(config):
                now = time.perf_counter()
                kind = event.get("type")
                if kind == "progress_start":
                    opened[event["stage"]] = now
                elif kind == "progress_end" and event["stage"] in opened:
                    stages[event["stage"]] = now - opened.pop(event["stage"])
                elif kind == "finish":
                    return event["translate_result"]
                elif kind == "error":
                    raise RuntimeError(event.get("error"))

        result = asyncio.run(drive())
        return result.mono_pdf_path, stages

    run.inputs = []
    return run
