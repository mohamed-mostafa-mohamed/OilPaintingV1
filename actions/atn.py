"""Reader for Photoshop action files (.atn, version 16).

An .atn file is a recording: a list of actions, each a list of steps. Every
step is a Photoshop "event" (what was done) plus a descriptor (all of its
settings). This module turns that binary recording into plain Python data:

    ActionSet(name, actions=[Action(name, steps=[Step(event, title, enabled,
                                                     dialog, desc)])])

Descriptors become dicts. Typed values keep their type where it matters:
    UnitFloat(unit, value)  - "#Pxl" 4.5, "#Prc" 60.0, "#Ang" 120 ...
    Enum(type, value)       - ("BlnM", "Ovrl") ...
    Ref([...])              - a reference to a layer, channel, document ...
    Obj(cls, {...})         - a nested descriptor with its class id
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field


class AtnError(ValueError):
    pass


@dataclass
class UnitFloat:
    unit: str
    value: float

    def __repr__(self):
        return f"{self.value:g}{_UNIT_SUFFIX.get(self.unit, ' ' + self.unit)}"


_UNIT_SUFFIX = {"#Pxl": "px", "#Prc": "%", "#Ang": "°", "#Rsl": "dpi", "#Rlt": "pt", "#Nne": "", "#Pnt": "pt"}


@dataclass
class Enum:
    type: str
    value: str

    def __repr__(self):
        return self.value


@dataclass
class Obj:
    cls: str
    items: dict
    name: str = ""

    def get(self, key, default=None):
        return self.items.get(key, default)

    def __getitem__(self, key):
        return self.items[key]

    def __contains__(self, key):
        return key in self.items

    def __repr__(self):
        return f"{self.cls}{self.items!r}"


@dataclass
class RefItem:
    form: str            # prop, Clss, Enmr, rele, Idnt, indx, name
    cls: str
    value: object = None


@dataclass
class Ref:
    items: list

    def first(self):
        return self.items[0] if self.items else None

    def __repr__(self):
        return "Ref(" + ", ".join(f"{i.cls}:{i.form}={i.value!r}" for i in self.items) + ")"


@dataclass
class Step:
    event: str           # "slct", "Mk  ", "gaussianBlur", "setd" ...
    title: str           # what the Actions panel shows: "Select", "Make" ...
    enabled: bool
    dialog: bool         # step was set to show its dialog
    desc: Obj | None

    def get(self, key, default=None):
        return self.desc.items.get(key, default) if self.desc else default


@dataclass
class Action:
    name: str
    steps: list = field(default_factory=list)


@dataclass
class ActionSet:
    name: str
    actions: list = field(default_factory=list)

    def action(self, name=None):
        if name is None:
            return self.actions[0] if self.actions else None
        for a in self.actions:
            if a.name == name:
                return a
        low = name.strip().lower()
        for a in self.actions:
            if a.name.strip().lower() == low:
                return a
        return None


class _Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.o = 0

    def take(self, n):
        if self.o + n > len(self.d):
            raise AtnError(f"file ends early at byte {self.o}")
        b = self.d[self.o:self.o + n]
        self.o += n
        return b

    def u8(self):
        return self.take(1)[0]

    def u16(self):
        return struct.unpack(">H", self.take(2))[0]

    def u32(self):
        return struct.unpack(">I", self.take(4))[0]

    def i32(self):
        return struct.unpack(">i", self.take(4))[0]

    def i64(self):
        return struct.unpack(">q", self.take(8))[0]

    def f64(self):
        return struct.unpack(">d", self.take(8))[0]

    def code(self):
        return self.take(4).decode("latin-1")

    def ustr(self):
        n = self.u32()
        if n > 1_000_000:
            raise AtnError("bad string length")
        return self.take(2 * n).decode("utf-16-be", "replace").rstrip("\x00")

    def pstr(self):
        n = self.u32()
        if n > 1_000_000:
            raise AtnError("bad string length")
        return self.take(n).decode("latin-1")

    def key(self):
        """A class/key id: length 0 means a 4-char code follows."""
        n = self.u32()
        if n == 0:
            return self.code()
        if n > 10_000:
            raise AtnError("bad id length")
        return self.take(n).decode("latin-1")

    # -- descriptors ----------------------------------------------------
    def descriptor(self):
        name = self.ustr()
        cls = self.key()
        n = self.u32()
        items = {}
        for _ in range(n):
            k = self.key()
            items[k] = self.value(self.code())
        return Obj(cls, items, name)

    def value(self, t):
        if t in ("Objc", "GlbO"):
            return self.descriptor()
        if t == "obj ":
            return self.reference()
        if t == "VlLs":
            return [self.value(self.code()) for _ in range(self.u32())]
        if t == "doub":
            return self.f64()
        if t == "UntF":
            u = self.code()
            return UnitFloat(u, self.f64())
        if t == "UnFl":                     # list of unit floats
            u = self.code()
            return [UnitFloat(u, self.f64()) for _ in range(self.u32())]
        if t == "TEXT":
            return self.ustr()
        if t == "enum":
            return Enum(self.key(), self.key())
        if t == "long":
            return self.i32()
        if t == "comp":
            return self.i64()
        if t == "bool":
            return bool(self.u8())
        if t in ("type", "GlbC"):
            self.ustr()
            return ("class", self.key())
        if t in ("alis", "Pth ", "tdta"):
            n = self.u32()
            return ("data", t, self.take(n))
        if t == "ObAr":                     # object array (rare in actions)
            return ("ObAr", self.take_object_array())
        raise AtnError(f"unknown value type {t!r} at byte {self.o - 4}")

    def take_object_array(self):
        self.u32()                          # version
        name = self.ustr()
        cls = self.key()
        n = self.u32()
        items = {}
        for _ in range(n):
            k = self.key()
            t = self.code()
            items[k] = self.value(t)
        return Obj(cls, items, name)

    def reference(self):
        out = []
        for _ in range(self.u32()):
            form = self.code()
            if form == "prop":
                self.ustr()
                cls = self.key()
                out.append(RefItem("prop", cls, self.key()))
            elif form == "Clss":
                self.ustr()
                out.append(RefItem("Clss", self.key()))
            elif form == "Enmr":
                self.ustr()
                cls = self.key()
                out.append(RefItem("Enmr", cls, Enum(self.key(), self.key())))
            elif form == "rele":
                self.ustr()
                cls = self.key()
                out.append(RefItem("rele", cls, self.i32()))
            elif form in ("Idnt", "indx"):
                self.ustr()
                cls = self.key()
                out.append(RefItem(form, cls, self.i32()))
            elif form == "name":
                self.ustr()
                cls = self.key()
                out.append(RefItem("name", cls, self.ustr()))
            else:
                raise AtnError(f"unknown reference form {form!r}")
        return Ref(out)

    # -- the file -------------------------------------------------------
    def step(self):
        expanded = self.u8()
        enabled = self.u8()
        dialog = self.u8()
        self.u8()                           # dialog options
        kind = self.code()
        if kind == "TEXT":
            event = self.pstr()
        elif kind == "long":
            event = self.code()
        else:
            raise AtnError(f"bad step id {kind!r}")
        title = self.pstr()
        has = self.i32()
        desc = None
        if has == -1:
            desc = self.descriptor()
        elif has != 0:
            raise AtnError(f"bad descriptor flag {has}")
        del expanded
        return Step(event, title, bool(enabled), bool(dialog), desc)

    def action(self):
        self.u16()                          # function key
        self.u8()                           # shift
        self.u8()                           # command
        self.u16()                          # colour
        name = self.ustr()
        self.u8()                           # expanded
        n = self.u32()
        return Action(name, [self.step() for _ in range(n)])


def parse(data: bytes) -> ActionSet:
    r = _Reader(data)
    version = r.u32()
    if version != 16:
        raise AtnError(f"unsupported .atn version {version} (expected 16)")
    name = r.ustr()
    r.u8()
    n = r.u32()
    out = ActionSet(name)
    for _ in range(n):
        out.actions.append(r.action())
    return out


def load(path: str) -> ActionSet:
    with open(path, "rb") as fh:
        return parse(fh.read())
