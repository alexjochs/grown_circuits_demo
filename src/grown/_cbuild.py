"""Build small C kernels on first use and load them with ctypes.

A kernel's source is compiled with the system C compiler (`$CC`, default `cc`, `-O3`) into a cache directory
($GROWN_CACHE or ~/.cache/grown, one subdirectory per kernel) keyed by the source hash, so concurrent workers and
different source versions never collide. Loading returns None when no compiler is available or $GROWN_NO_C is set;
every caller has a pure numpy / Python fallback with identical results.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile

_LIBS: dict = {}


def cache_dir(name: str) -> str:
    d = os.environ.get("GROWN_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "grown")
    d = os.path.join(d, name)
    os.makedirs(d, exist_ok=True)
    return d


def load(name: str, src: str, signatures: dict, disable_env: str = "GROWN_NO_C", link=()):
    """Compile (once) and load `src`; `signatures` maps symbol -> (argtypes, restype); `link`: extra objects or
    libraries for the link line (part of the cache key only through `src`, so put their identity in the source).
    None if unavailable."""
    if name in _LIBS:
        return _LIBS[name]
    lib = None
    if not os.environ.get(disable_env) and not os.environ.get("GROWN_NO_C"):
        try:
            h = hashlib.sha1(src.encode()).hexdigest()[:16]
            path = os.path.join(cache_dir(name), f"{name}_{h}.so")
            if not os.path.exists(path):
                cc = os.environ.get("CC", "cc")
                with tempfile.TemporaryDirectory() as td:
                    cfile = os.path.join(td, "k.c")
                    with open(cfile, "w") as f:
                        f.write(src)
                    tmp = path + f".tmp{os.getpid()}"
                    subprocess.run([cc, "-O3", "-shared", "-fPIC", "-o", tmp, cfile, *link], check=True,
                                   capture_output=True)
                    os.replace(tmp, path)          # atomic: concurrent workers cannot load a torn file
            lib = ctypes.CDLL(path)
            for sym, (argtypes, restype) in signatures.items():
                fn = getattr(lib, sym)
                fn.argtypes = argtypes
                fn.restype = restype
        except Exception:          # no compiler, read-only cache, ...: callers fall back
            lib = None
    _LIBS[name] = lib
    return lib


def ptr(a) -> int:
    """Address of a numpy array's data (pass to a c_void_p argument)."""
    return a.__array_interface__["data"][0]
