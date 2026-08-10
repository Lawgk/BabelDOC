import copy
import dataclasses
from pathlib import Path

import orjson
from xsdata.formats.dataclass.context import XmlContext
from xsdata.formats.dataclass.parsers import XmlParser
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig

from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.frontend.il_creater_active_support import (
    LazyPassthroughInstruction,
)

# Matrix-valued attributes declared as `list[object]`. xsdata serialises them
# fine but has no type to parse them back into, so they return as strings and
# blow up the first time anything formats them as numbers.
_FLOAT_TOKEN_FIELDS = frozenset({"ctm", "relocation_transform"})


def _orjson_default(value):
    if isinstance(value, LazyPassthroughInstruction):
        return value.materialize()
    raise TypeError


def _walk(node, seen: set[int]):
    """Yield every dataclass instance reachable from node, once each."""
    if id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk(item, seen)
        return
    if not dataclasses.is_dataclass(node):
        return
    yield node
    for f in dataclasses.fields(node):
        value = getattr(node, f.name, None)
        if isinstance(value, (list, tuple)) or dataclasses.is_dataclass(value):
            yield from _walk(value, seen)


def _materialize_lazy(document: il_version_1.Document) -> None:
    """Resolve LazyPassthroughInstruction values before serialising.

    GraphicState.passthrough_per_char_instruction is typed `str` but carries a
    lazy object at runtime. to_json has a default hook for it; the XML
    serialiser has no such escape hatch and raises ConverterError instead —
    which is why write_xml could not emit a single byte for a real document.
    """
    for obj in _walk(document, set()):
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name, None)
            if isinstance(value, LazyPassthroughInstruction):
                setattr(obj, f.name, value.materialize())


def _restore_float_tokens(document: il_version_1.Document) -> None:
    """Undo the str-ification the parser applies to `list[object]` attributes."""
    for obj in _walk(document, set()):
        for name in _FLOAT_TOKEN_FIELDS:
            value = getattr(obj, name, None)
            if isinstance(value, list) and any(isinstance(v, str) for v in value):
                setattr(obj, name, [float(v) for v in value])


class XMLConverter:
    def __init__(self):
        self.parser = XmlParser()
        config = SerializerConfig(indent="  ")
        context = XmlContext()
        self.serializer = XmlSerializer(context=context, config=config)

    def write_xml(self, document: il_version_1.Document, path: str):
        with Path(path).open("w", encoding="utf-8") as f:
            f.write(self.to_xml(document))

    def read_xml(self, path: str) -> il_version_1.Document:
        with Path(path).open(encoding="utf-8") as f:
            return self.from_xml(f.read())

    def to_xml(self, document: il_version_1.Document) -> str:
        _materialize_lazy(document)
        return self.serializer.render(document)

    def from_xml(self, xml: str) -> il_version_1.Document:
        document = self.parser.from_string(
            xml,
            il_version_1.Document,
        )
        _restore_float_tokens(document)
        return document

    def deepcopy(self, document: il_version_1.Document) -> il_version_1.Document:
        return copy.deepcopy(document)
        # return self.from_xml(self.to_xml(document))

    def to_json(self, document: il_version_1.Document) -> str:
        return orjson.dumps(
            document,
            option=orjson.OPT_APPEND_NEWLINE
            | orjson.OPT_INDENT_2
            | orjson.OPT_SORT_KEYS,
            default=_orjson_default,
        ).decode()

    def write_json(self, document: il_version_1.Document, path: str):
        with Path(path).open("w", encoding="utf-8") as f:
            f.write(self.to_json(document))
