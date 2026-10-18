"""
Make a protected copy of OilStudio to send to clients - no readable code.

    python build/protect.py                  -> .release/  (then PyInstaller)
    python build/protect.py --out DIR

What it does, on a copy (the project itself is never touched):

  1. JavaScript and CSS are minified: comments and formatting removed, and
     with Node.js installed the JavaScript variable names are scrambled too
     (terser).  Without Node, rjsmin/rcssmin strip comments and whitespace.
  2. The templates, scripts and styles are packed, compressed, into a
     Python module (_oilstudio_assets) - no loose files to open.
  3. Every Python module - the server, the painting engine, the planner,
     the action player, the packed files - is compiled to machine code
     with Cython (.pyd on Windows, .so on macOS/Linux).  All .py source and
     bytecode is removed; a two-line launcher (OilStudio.py) starts the
     compiled app.
  4. The compiled app is imported and smoke-tested, and the imports it needs
     are written to build/imports.json for PyInstaller
     (build/OilStudio-protected.spec).

Needs: Cython and setuptools (pip), and a C compiler -
  Windows: "Build Tools for Visual Studio", workload "Desktop development
           with C++" (free)
  macOS:   Xcode Command Line Tools  (xcode-select --install)
Optional: Node.js (better JavaScript protection), rjsmin and rcssmin (pip).

--no-compile does everything except step 3 (to check the result quickly).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# what goes into the release
CODE = ["app.py", "desktop.py", "oilpaint", "actions"]
DATA = ["samples", "ps_actions", "requirements.txt"]
EMBED = ["templates", "static"]
ICONS = ["build/icon.ico", "build/icon.icns"]
LAUNCHER = "OilStudio.py"
ASSETS_MODULE = "_oilstudio_assets"

# compiler settings: plain Python semantics, keyword arguments everywhere
# (Flask calls views with keywords), annotations are not C types
DIRECTIVES = {
    "language_level": 3,
    "annotation_typing": False,
    "always_allow_keywords": True,
    "binding": True,
    "embedsignature": False,
    "emit_code_comments": False,
}


def say(msg):
    print(f"== {msg}", flush=True)


def warn(msg):
    print(f"!! {msg}", flush=True)


# --------------------------------------------------------------------------
# 1. copy
# --------------------------------------------------------------------------


def copy_tree(src, dst):
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store")
    if os.path.isdir(src):
        shutil.copytree(src, dst, ignore=ignore)
    elif os.path.isfile(src):
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src, dst)


def stage(out):
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for name in CODE + DATA + EMBED + ICONS:
        src = os.path.join(ROOT, name)
        if os.path.exists(src):
            copy_tree(src, os.path.join(out, name))
    # the spec file that packs the protected app
    spec = os.path.join(ROOT, "build", "OilStudio-protected.spec")
    shutil.copy2(spec, os.path.join(out, "OilStudio-protected.spec"))


# --------------------------------------------------------------------------
# 2. minify the browser code
# --------------------------------------------------------------------------


def _npx():
    for name in ("npx.cmd", "npx") if os.name == "nt" else ("npx",):
        path = shutil.which(name)
        if path:
            return path
    return None


USE_NODE = True


def minify_js(path):
    npx = _npx() if USE_NODE else None
    if npx:
        tmp = path + ".min"
        try:
            r = subprocess.run([npx, "--yes", "terser@5", path, "--compress", "--mangle",
                                "--comments", "false", "--output", tmp],
                               capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            r = subprocess.CompletedProcess([], 1, "", "terser did not finish in 5 minutes")
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, path)
            return "terser (names scrambled)"
        warn(f"terser failed on {os.path.basename(path)}: {r.stderr.strip()[:300]}")
        if os.path.exists(tmp):
            os.remove(tmp)
    try:
        import rjsmin
    except ImportError:
        return None
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(rjsmin.jsmin(src))
    return "rjsmin"


def minify_css(path):
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    try:
        import rcssmin
        out = rcssmin.cssmin(src)
    except ImportError:
        out = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        out = "\n".join(line.strip() for line in out.splitlines() if line.strip())
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)


def minify(out):
    used = set()
    for folder, _, files in os.walk(os.path.join(out, "static")):
        for f in files:
            p = os.path.join(folder, f)
            if f.endswith(".js"):
                how = minify_js(p)
                used.add(how or "none")
            elif f.endswith(".css"):
                minify_css(p)
    # HTML comments in the templates (Jinja comments never reach the page)
    for folder, _, files in os.walk(os.path.join(out, "templates")):
        for f in files:
            p = os.path.join(folder, f)
            with open(p, encoding="utf-8") as fh:
                html = fh.read()
            html = re.sub(r"<!--(?!\[).*?-->", "", html, flags=re.S)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(html)
    if "none" in used:
        warn("JavaScript was NOT minified: install Node.js (best) or  pip install rjsmin rcssmin")
    else:
        say("JavaScript minified with " + ", ".join(sorted(used)))


# --------------------------------------------------------------------------
# 3. pack templates and static files into a module
# --------------------------------------------------------------------------


def embed(out):
    def collect(base):
        items = {}
        root = os.path.join(out, base)
        for folder, _, files in os.walk(root):
            for f in files:
                p = os.path.join(folder, f)
                rel = os.path.relpath(p, root).replace(os.sep, "/")
                with open(p, "rb") as fh:
                    items[rel] = zlib.compress(fh.read(), 9)
        return items

    templates, static = collect("templates"), collect("static")

    # small pieces: the Windows C compiler refuses text constants over ~16 KB
    def pieces(blob, size=2000):
        return "(" + "".join(f"{blob[i:i + size]!r}, " for i in range(0, len(blob), size)) + ")"

    lines = [
        '"""Templates, scripts and styles of OilStudio, packed (generated by',
        'build/protect.py - do not edit)."""',
        "import zlib",
        f"VERSION = {int(time.time())}",
        "_T = {",
    ]
    lines += [f"    {k!r}: {pieces(v)}," for k, v in sorted(templates.items())]
    lines += ["}", "_S = {"]
    lines += [f"    {k!r}: {pieces(v)}," for k, v in sorted(static.items())]
    lines += [
        "}",
        "_cache = {}",
        "",
        "",
        "def templates():",
        "    return {k: zlib.decompress(b''.join(v)).decode('utf-8') for k, v in _T.items()}",
        "",
        "",
        "def static(name):",
        "    if name not in _S:",
        "        return None",
        "    if name not in _cache:",
        "        _cache[name] = zlib.decompress(b''.join(_S[name]))",
        "    return _cache[name]",
        "",
    ]
    with open(os.path.join(out, ASSETS_MODULE + ".py"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    for base in EMBED:
        shutil.rmtree(os.path.join(out, base))
    say(f"Packed {len(templates)} template(s) and {len(static)} static file(s) into the program")


# --------------------------------------------------------------------------
# 4. compile
# --------------------------------------------------------------------------


def split_package_inits(out):
    """A package __init__ with real code moves to _init_impl.py (compiled);
    the __init__.py left behind only re-exports it."""
    for pkg in ("oilpaint", "actions"):
        init = os.path.join(out, pkg, "__init__.py")
        with open(init, encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        trivial = all(isinstance(n, (ast.Expr, ast.Assign)) for n in tree.body)
        if trivial:
            continue
        with open(os.path.join(out, pkg, "_init_impl.py"), "w", encoding="utf-8") as fh:
            fh.write(src)
        names = []
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in n.targets):
                names = [ast.literal_eval(e) for e in n.value.elts]
        with open(init, "w", encoding="utf-8") as fh:
            fh.write("from ._init_impl import *  # noqa: F401,F403\n")
            if names:
                fh.write(f"from ._init_impl import {', '.join(names)}  # noqa: F401\n")
                fh.write(f"__all__ = {names!r}\n")


def modules_to_compile(out):
    mods = []
    for folder, _, files in os.walk(out):
        rel = os.path.relpath(folder, out)
        if rel.split(os.sep)[0] in ("samples", "ps_actions", "build"):
            continue
        for f in files:
            if not f.endswith(".py") or f == "__init__.py":
                continue
            if rel == "." and f in (LAUNCHER,) or f.endswith(".spec"):
                continue
            path = os.path.normpath(os.path.join(rel, f))
            name = path[:-3].replace(os.sep, ".")
            mods.append((name, path))
    return sorted(mods)


def scan_imports(out):
    """Every module the compiled code imports (PyInstaller cannot read
    imports inside compiled files, so it is told about them)."""
    found, ours = set(), set()
    for name, path in modules_to_compile(out):
        ours.add(name)
    for folder, _, files in os.walk(out):
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(folder, f)
            rel = os.path.relpath(path, out)[:-3].replace(os.sep, ".")
            pkg = rel.rsplit(".", 1)[0] if "." in rel else ""
            if f == "__init__.py":
                pkg = rel.rsplit(".", 1)[0]
                ours.add(pkg)
            with open(path, encoding="utf-8") as fh:
                try:
                    tree = ast.parse(fh.read())
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        found.add(a.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        base = pkg
                        for _ in range(node.level - 1):
                            base = base.rsplit(".", 1)[0] if "." in base else ""
                        mod = ".".join(x for x in (base, node.module) if x)
                        for a in node.names:
                            found.add(f"{mod}.{a.name}" if mod else a.name)
                        found.add(mod)
                    elif node.module:
                        found.add(node.module)
                        for a in node.names:
                            found.add(f"{node.module}.{a.name}")
    found |= ours
    # keep real module names only (a "from x import function" adds x.function)
    keep = set()
    for m in found:
        if not m or m.startswith("_init_impl"):
            continue
        keep.add(m)
    return sorted(keep), sorted(ours)


def compile_all(out):
    try:
        from Cython.Build import cythonize
        from setuptools import Extension, setup
    except ImportError:
        sys.exit("Cython is not installed:  pip install cython setuptools")
    mods = modules_to_compile(out)
    say(f"Compiling {len(mods)} Python modules to machine code (a few minutes)")
    exts = [Extension(name, [path]) for name, path in mods]
    here = os.getcwd()
    os.chdir(out)
    try:
        ext_modules = cythonize(exts, compiler_directives=DIRECTIVES, quiet=True,
                                nthreads=0 if os.name == "nt" else (os.cpu_count() or 1))
        try:
            setup(name="oilstudio", ext_modules=ext_modules,
                  script_args=["build_ext", "--inplace", "--build-temp", "_cbuild"])
        except SystemExit as exc:
            if exc.code not in (0, None):
                raise
    except Exception as exc:                          # noqa: BLE001
        hint = ("Install 'Build Tools for Visual Studio' with the workload "
                "'Desktop development with C++', then run the build again."
                if os.name == "nt" else
                "Install the Xcode Command Line Tools:  xcode-select --install")
        sys.exit(f"Compiling failed: {exc}\n{hint}")
    finally:
        os.chdir(here)
    # remove the source and everything the compiler left behind
    for name, path in mods:
        base = os.path.join(out, path[:-3])
        for ext in (".py", ".c"):
            if os.path.exists(base + ext):
                os.remove(base + ext)
        if not any(f.startswith(os.path.basename(base) + ".") and f.endswith((".pyd", ".so"))
                   for f in os.listdir(os.path.dirname(base) or out)):
            sys.exit(f"Compiling failed: no compiled file for {name}")
    for junk in ("_cbuild", "build/lib", "oilstudio.egg-info"):
        p = os.path.join(out, junk)
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)


# --------------------------------------------------------------------------
# 5. launcher, checks
# --------------------------------------------------------------------------


def write_launcher(out):
    with open(os.path.join(out, LAUNCHER), "w", encoding="utf-8") as fh:
        fh.write("import desktop\n\ndesktop.run()\n")


def smoke_test(out):
    """Import the compiled app in a fresh Python and fetch the main page and
    a script - the build stops here if anything is wrong."""
    code = (
        "import os, sys; sys.path.insert(0, os.getcwd());"
        "import app;"
        "c = app.app.test_client();"
        "r = c.get('/'); assert r.status_code == 200, r.status_code;"
        "html = r.get_data(as_text=True);"
        "import re; js = re.findall(r'/static/js/[a-z]+\\.js[^\"]*', html);"
        "assert js, 'no scripts linked';"
        "assert all(c.get(u).status_code == 200 for u in js), 'a script is missing';"
        "from oilpaint import planner, engine, params;"
        "import numpy as np;"
        "img = (np.random.default_rng(1).random((90, 120, 3)) * 255).astype('uint8');"
        "plan = planner.plan_preview(img, params.defaults()); assert plan.count > 0;"
        "print('ok', len(js), 'scripts,', plan.count, 'strokes')"
    )
    env = dict(os.environ, OILSTUDIO_WORKDIR=os.path.join(out, "_smoke"))
    r = subprocess.run([sys.executable, "-c", code], cwd=out, capture_output=True,
                       text=True, env=env)
    shutil.rmtree(os.path.join(out, "_smoke"), ignore_errors=True)
    if r.returncode != 0:
        sys.exit("The protected app does not start:\n" + (r.stderr or r.stdout)[-2500:])
    say("Smoke test passed: " + r.stdout.strip())


def leftover_source(out):
    left = []
    for folder, _, files in os.walk(out):
        rel = os.path.relpath(folder, out)
        if rel.split(os.sep)[0] in ("samples", "ps_actions"):
            continue
        for f in files:
            if f.endswith((".py", ".pyc", ".c")) and f not in (LAUNCHER, "__init__.py"):
                left.append(os.path.join(rel, f))
    return left


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, ".release"))
    ap.add_argument("--no-compile", action="store_true",
                    help="skip compiling (quick check of the rest)")
    ap.add_argument("--no-node", action="store_true",
                    help="do not use Node.js/terser for the JavaScript")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    global USE_NODE
    USE_NODE = not args.no_node

    say(f"Protected copy in {out}")
    stage(out)
    minify(out)
    embed(out)
    split_package_inits(out)
    write_launcher(out)
    imports, ours = scan_imports(out)
    os.makedirs(os.path.join(out, "build"), exist_ok=True)
    with open(os.path.join(out, "build", "imports.json"), "w", encoding="utf-8") as fh:
        json.dump({"imports": imports, "ours": ours}, fh, indent=1)
    if not args.no_compile:
        compile_all(out)
        left = leftover_source(out)
        if left:
            sys.exit("Source files are still in the release: " + ", ".join(left))
    smoke_test(out)
    for folder, dirs, _ in os.walk(out):
        for d in list(dirs):
            if d == "__pycache__":
                shutil.rmtree(os.path.join(folder, d), ignore_errors=True)
    say("Done" + (" (NOT compiled: --no-compile)" if args.no_compile else
                  " - no Python source, no loose templates or scripts"))


if __name__ == "__main__":
    main()
