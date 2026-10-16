"""serve/runconfig.py - #564: the web page's Settings view of the run config (strata-<model>.json).

A short list of documented keys can be read and changed from Strata's own page (GET / POST /config).  Everything
else in the file - the keys setup writes, the network and security keys (host, api_key, cors_origins,
trusted_origins, allowed_hosts), the MCP servers and before_load (which run programs), sampling keys not listed
here - is kept as it is: a change touches only the keys it names (as setup run again keeps the user's keys, #629).
The earlier file is kept as strata-<model>.json.bak.  The server reads the config when it starts, so a change is
used from the next start on.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

from serve import engine_options

# engine_args() (serve/server.py) synthesizes these catalog flags from higher-level config keys: a config with
# "vram_elastic": true runs --vram-elastic with no flag in "args", "parallel": N runs --batch N, and so on.  The
# Engine tab reads the effective spawn (engine_args), so clearing a flag there must clear the key that writes it.
SYNTHESIZED = {"argflag.--vram-elastic": ("vram_elastic", "vram_segment_mib"),
               "arg.--vram-segment-mib": ("vram_segment_mib",),
               "arg.--batch": ("parallel",),
               "arg.--layer-split": ("layer_split",)}

# (key, kind, help).  kind: "bool", "int>=0", "num>=0", ("enum", values), "names", or ("sampling", check) for a key
# of the "sampling" block, ("arg", flag) for an engine option kept in "args".
# The Engine tab's keys - "env.NAME", "arg.--flag" (int), "argflag.--flag" (presence), "argstr.--flag",
# "argenum.--flag" - are not listed here: serve/engine_options.py's catalog is their allowlist.
EDITABLE = [
    ("sampling.temperature", ("sampling", "num>=0"),
     "Default temperature for requests that send none (0 = greedy, the default without a sampling block)"),
    ("sampling.top_p", ("sampling", "0<x<=1"), "Default top_p for requests that send none"),
    ("sampling.top_k", ("sampling", "1..64"), "Default top_k for requests that send none (1-64)"),
    ("sampling.min_p", ("sampling", "0<=x<=1"), "Default min_p for requests that send none"),
    ("parallel", "int>=0", "Up to this many requests decode together in the engine's batch slots (1 or empty: one at a time, #465)"),
    ("reasoning_budget_tokens", "int>=0", "Cap the thinking of every request at this many tokens (0 or empty: no cap)"),
    ("fit_max_tokens", "bool", "Shorten a max_tokens that does not fit the context instead of answering 400"),
    ("anthropic_thinking", ("enum", ["model", "on_request"]),
     "Anthropic requests that do not ask for thinking: think as the model does (model) or not (on_request)"),
    ("effort_position", ("enum", ["start", "end"]),
     "Where a non-default reasoning effort goes: start (the default) or end (keeps the cache when it changes)"),
    ("aliases", "names", "Other model names the server lists and answers to (comma-separated)"),
    ("idle_unload_s", "num>=0", "Unload the model after this many seconds without requests (0 or empty: never)"),
    ("lazy_load", "bool", "Start without loading the model; the first request loads it (text only)"),
    ("engine_silence_s", "num>=0", "End a request when the engine says nothing for this long (default 300 s, 0 = wait)"),
    ("api_monitor", "bool", "Keep the last 100 requests' prompts and answers in memory for /api-monitor"),
    ("open_browser", "bool", "Open the chat page in the browser when the model is ready"),
    ("vram_reserve_mib", ("arg", "--vram-reserve-mib"),
     "VRAM in MiB the engine leaves free for other programs (engine default 700)"),
    # three everyday engine flags the generated catalog does not carry (they are not GPU speed switches);
    # stored in "args" the same way vram_reserve_mib is.  "argnum" = a float arg, "argchoice" = a fixed set.
    ("conversation_cache_mib", ("arg", "--conversation-cache-mib"),
     "RAM in MiB kept for parked conversations so coming back does not re-read them (0 or empty: off)"),
    ("pcie_frac", ("argnum", "--pcie-frac"),
     "Share of the CPU-to-GPU link the model may use to pull experts from RAM (0.05-1; lower leaves more for everything else)"),
    ("ple_io", ("argchoice", "--ple-io", ("direct", "ram", "mmap")),
     "How the big n-gram lookup table is read: direct is made for SSDs, ram keeps it in RAM, mmap lets the OS cache it"),
]
SPEC = {k: kind for k, kind, _ in EDITABLE}

# The Engine tab's Simple settings (#2509): everyday knobs in plain words, each one a key EDITABLE or the
# generated catalog already validates - check() is the same allowlist, the web layer only renders this.
# (key, label, control, one plain sentence).  control: ("select", values), ("number",) or ("checkbox",).
# Keys the catalog does not carry (--pcie-frac, --ple-io, --conversation-cache-mib) are not listed here.
SIMPLE = [
    ("parallel", "Conversations at once", ("select", [1, 2, 4, 8]),
     "How many requests run side by side. Each busy conversation costs RAM; 1 is fine for one person."),
    ("arg.--spec", "Guess-ahead speedup", ("select", [2, 4, 6, 8]),
     "The model guesses several tokens ahead and verifies them in one pass. Higher guesses more; the default is tuned."),
    ("vram_reserve_mib", "VRAM left free for other programs", ("number",),
     "VRAM in MiB the engine leaves free for other programs (engine default 700)"),
    ("conversation_cache_mib", "Switch between conversations instantly", ("select", [2048, 4096, 8192]),
     "Keeps parked conversations in RAM so coming back to one does not re-read it. Costs RAM up to this size."),
    ("pcie_frac", "PCIe share for model streaming", ("number",),
     "How much of the CPU-to-GPU link the model may use to pull experts from RAM. Lower leaves more for everything else."),
    ("ple_io", "Model lookup table reads", ("select", ["direct", "ram", "mmap"]),
     "How the big n-gram lookup table is read. direct is made for SSDs; ram keeps it in RAM; mmap lets the OS cache it."),
    ("idle_unload_s", "Unload the model when idle", ("number",),
     "Unload the model after this many seconds without requests; 0 or empty: never unload."),
    ("reasoning_budget_tokens", "Cap the thinking per request", ("number",),
     "Cap the thinking of every request at this many tokens (0 or empty: no cap)"),
    ("fit_max_tokens", "Shorten over-long answers instead of refusing", ("checkbox",),
     "A request that asks for more than the context fits gets its answer shortened to the room left, not refused."),
    ("open_browser", "Open the chat page when the model is ready", ("checkbox",),
     "The server opens this page in your browser once the model has loaded."),
]


def _arg(cfg: dict, flag: str):
    a = cfg.get("args") if isinstance(cfg.get("args"), list) else []
    if flag in a[:-1]:
        v = str(a[a.index(flag) + 1])
        return int(v) if v.isdigit() else v
    return None


def value_of(cfg: dict, key: str):
    if key in engine_options.CATALOG:
        kind, _, rest = key.partition(".")
        if kind == "env":
            e = cfg.get("env")
            return e.get(rest) if isinstance(e, dict) else None
        flag = engine_options.CATALOG[key]["flag"]
        a = cfg.get("args")
        if not isinstance(a, list):
            a = []
        if kind == "argflag":
            return flag in a
        if flag in a[:-1]:
            v = str(a[a.index(flag) + 1])
            if kind == "arg" and v.isdigit():
                return int(v)
            if engine_options.CATALOG[key].get("kind") == "number":
                try:
                    return float(v)
                except ValueError:
                    pass
            return v
        return None
    kind = SPEC[key]
    if isinstance(kind, tuple) and kind[0] == "sampling":
        s = cfg.get("sampling")
        return s.get(key.split(".", 1)[1]) if isinstance(s, dict) else None
    if isinstance(kind, tuple) and kind[0] in ("arg", "argnum", "argchoice"):
        return _arg(cfg, kind[1])
    return cfg.get(key)


def view(cfg: dict, path: str | Path) -> dict:
    """GET /config: the editable keys with their values (None: not set, the default applies), and the
    Engine tab's Simple settings with theirs."""
    out = []
    for key, kind, help_ in EDITABLE:
        k = kind[0] if isinstance(kind, tuple) else kind
        out.append({"key": key, "value": value_of(cfg, key), "help": help_,
                    "kind": "number" if k in ("sampling", "arg", "argnum", "int>=0", "num>=0")
                    else ("enum" if k == "argchoice" else k),
                    **({"choices": kind[2] if k == "argchoice" else kind[1]} if k in ("enum", "argchoice") else {})})
    simple = [{"key": key, "label": label, "desc": desc, "value": value_of(cfg, key), "kind": control[0],
               **({"choices": list(control[1])} if control[0] == "select" else {})}
              for key, label, control, desc in SIMPLE]
    return {"file": Path(path).name, "keys": out, "simple": simple,
            "note": "Saved to the run config; used from the next start of the model."}


def _number(key, v, whole=False, lo=0.0, hi=None, lo_open=False):
    # math.isfinite: json.loads lets NaN/Infinity through a bare number too (1e400); int(inf) would raise
    # OverflowError here and kill the handler thread - a plain ValueError answers 400 instead.
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or \
            (whole and v != int(v)) or \
            (v <= lo if lo_open else v < lo) or (hi is not None and v > hi):
        rng = f"{'more than' if lo_open else 'at least'} {lo:g}" + (f" and at most {hi:g}" if hi is not None else "")
        raise ValueError(f"{key}: expected a {'whole ' if whole else ''}number, {rng}, not {v!r}")
    return int(v) if whole else v


def check(key: str, v, cfg: dict):
    """The value to store for `key`, or a ValueError naming what is expected.  None removes the key (its default)."""
    if key in engine_options.CATALOG:
        if v is None:
            return None
        kind, _, rest = key.partition(".")
        o = engine_options.CATALOG[key]
        input_kind = o.get("kind", "str")
        if kind == "argflag":
            if not isinstance(v, bool):
                raise ValueError(f"{key}: expected true or false, not {v!r}")
            return v
        if kind == "arg" and input_kind == "int":
            return _number(key, v, whole=True, lo=o.get("min", 0), hi=o.get("max"))
        if kind == "arg" and input_kind == "number":
            return _number(key, v, lo=o.get("min", 0), hi=o.get("max"))
        if kind == "argenum":
            if v not in o.get("choices", ()):
                raise ValueError(f"{key}: expected one of {', '.join(o['choices'])}, not {v!r}")
            return v
        # env.NAME and string args: the engine parses the string itself
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str):
            s = v.strip()
            if not s or len(s) > 512 or any(ord(c) < 32 or c == "\x7f" for c in s):
                raise ValueError(f"{key}: expected a short value with no control characters, not {v!r}")
            return s
        raise ValueError(f"{key}: expected a value, not {v!r}")
    if key not in SPEC:
        raise ValueError(f"{key!r} cannot be changed here (only the keys the Settings view lists)")
    if v is None:
        return None
    kind = SPEC[key]
    k = kind[0] if isinstance(kind, tuple) else kind
    if k == "bool":
        if not isinstance(v, bool):
            raise ValueError(f"{key}: expected true or false, not {v!r}")
        if key == "lazy_load" and v and cfg.get("vision"):
            raise ValueError("lazy_load: lazy loading is text-only, and this model reads images (\"vision\")")
        return v
    if k == "enum":
        if v not in kind[1]:
            raise ValueError(f"{key}: expected one of {', '.join(kind[1])}, not {v!r}")
        return v
    if k == "names":
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() and len(x) <= 128 for x in v):
            raise ValueError(f"{key}: expected a list of model names, not {v!r}")
        return v or None
    if k == "int>=0":
        return _number(key, v, whole=True)
    if k == "num>=0":
        return _number(key, v)
    if k == "arg":
        return _number(key, v, whole=True)
    if k == "argnum":
        return _number(key, v, lo=0.05, hi=1.0)
    if k == "argchoice":
        if v not in kind[2]:
            raise ValueError(f"{key}: expected one of {', '.join(kind[2])}, not {v!r}")
        return v
    rule = kind[1]                                     # a sampling key: the server's own rules
    if rule == "num>=0":
        return float(_number(key, v))
    if rule == "0<x<=1":
        return float(_number(key, v, hi=1, lo_open=True))
    if rule == "0<=x<=1":
        return float(_number(key, v, hi=1))
    return _number(key, v, whole=True, lo=1, hi=64)


def apply(cfg: dict, changes: dict) -> tuple[dict, list[str]]:
    """The config with `changes` ({key: value or None}) made, and the keys that changed.  Every value is checked
    first, so a request with one bad value changes nothing (ValueError)."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError('send {"set": {"<key>": <value or null>, ...}}')
    checked = {k: check(k, v, cfg) for k, v in changes.items()}
    new = json.loads(json.dumps(cfg))                  # a deep copy: the caller's dict is left alone
    changed = []
    for key, v in checked.items():
        kind = key.partition(".")[0]
        # engine_args() synthesizes some catalog flags from higher-level keys; clearing the flag must clear
        # the key that writes it, so the effective spawn matches what the Engine tab shows and resets do.
        clears_synth = (key in SYNTHESIZED and (v is None or (kind == "argflag" and v is False))
                        and any(k in new for k in SYNTHESIZED[key]))
        if value_of(new, key) == v and not clears_synth:
            continue
        if key in engine_options.CATALOG:
            kind, _, rest = key.partition(".")
            if kind == "env":
                e = new.get("env")
                e = dict(e) if isinstance(e, dict) else {}
                if v is None:
                    e.pop(rest, None)
                else:
                    e[rest] = v
                if e:
                    new["env"] = e
                else:
                    new.pop("env", None)
            else:
                a = new.get("args")
                a = list(a) if isinstance(a, list) else []
                flag = engine_options.CATALOG[key]["flag"]
                if kind == "argflag":
                    if v and flag not in a:
                        a.append(flag)
                    elif not v and flag in a:
                        a.remove(flag)
                elif flag in a[:-1]:
                    i = a.index(flag)
                    if v is None:
                        del a[i:i + 2]
                    else:
                        a[i + 1] = str(v)
                elif v is not None:
                    a += [flag, str(v)]
                new["args"] = a
            if clears_synth:
                for cfg_key in SYNTHESIZED[key]:
                    new.pop(cfg_key, None)
            changed.append(key)
            continue
        kind = SPEC[key]
        if isinstance(kind, tuple) and kind[0] == "sampling":
            s = dict(new.get("sampling") or {}) if isinstance(new.get("sampling"), dict) else {}
            sub = key.split(".", 1)[1]
            if v is None:
                s.pop(sub, None)
            else:
                s[sub] = v
            if s:
                new["sampling"] = s
            else:
                new.pop("sampling", None)
        elif isinstance(kind, tuple) and kind[0] in ("arg", "argnum", "argchoice"):
            a = list(new.get("args") or [])
            flag = kind[1]
            if flag in a[:-1]:
                i = a.index(flag)
                if v is None:
                    del a[i:i + 2]
                else:
                    a[i + 1] = str(v)
            elif v is not None:
                a += [flag, str(v)]
            new["args"] = a
        elif v is None:
            new.pop(key, None)
        else:
            new[key] = v
        changed.append(key)
    return new, changed


def backup_of(path: str | Path) -> Path:
    """The .bak a save keeps for the config at `path`."""
    return Path(path).with_name(Path(path).name + ".bak")


def save(path: str | Path, cfg: dict) -> Path:
    """The config written whole (a temporary file moved over the old one), the earlier one kept as <name>.bak."""
    path = Path(path)
    bak = backup_of(path)
    shutil.copyfile(path, bak)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)     # a locked target (Windows WinError 5) must not leave .tmp litter
        raise
    return bak


def restore(path: str | Path) -> Path:
    """Undo the last save: the <name>.bak the save kept is copied back over the config (the .bak stays, so the
    same earlier file can be restored again).  ValueError from load() when the .bak is not a JSON object."""
    path = Path(path)
    bak = backup_of(path)
    load(bak)                           # refuse to put back a file the server could not read
    shutil.copyfile(bak, path)
    return bak


def load(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{Path(path).name} is not a JSON object")
    return cfg
