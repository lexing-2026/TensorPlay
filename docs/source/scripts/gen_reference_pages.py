#!/usr/bin/env python3
"""Generate TensorPlay reference pages from the bundled source pages.

The script parses each source page for section headings, autosummary entries,
and active currentmodule directives; resolves every entry against the
corresponding TensorPlay module; and emits pages containing the symbols that
exist today. Missing entries are dropped, and public TensorPlay-only symbols
are appended as an explicit additions section.
"""
import inspect
import os
import re
import sys

TP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REFERENCE_SOURCE = os.path.join(TP_ROOT, "third_party", "pytorch", "docs", "source")
OUT = os.path.join(TP_ROOT, "docs", "source")

def map_module(name):
    if name == "torch":
        return "tensorplay"
    if name.startswith("torch.ao."):
        return "tensorplay.quantization"
    if name.startswith("torch."):
        return "tensorplay." + name[len("torch."):]
    return None


MODULE_MAP = {k: map_module(k) for k in [
    "torch", "torch.autograd", "torch.nn", "torch.nn.functional",
    "torch.nn.init", "torch.optim", "torch.optim.lr_scheduler",
    "torch.optim.swa_utils", "torch.cuda", "torch.amp", "torch.linalg",
    "torch.fft", "torch.special", "torch.sparse", "torch.random",
    "torch.utils.data", "torch.utils", "torch.futures", "torch.hub",
    "torch.multiprocessing", "torch.library", "torch.distributed",
    "torch.quantization",
]}

sys.path.insert(0, TP_ROOT)

HIDDEN_ROLE = (
    "```{eval-rst}\n.. role:: hidden\n    :class: hidden-section\n```\n"
)


def resolve(mod_name, entry):
    """Resolve dotted entry inside mapped tensorplay module; return obj or None."""
    import importlib
    cand_parts = []
    if mod_name is not None:
        m = map_module(mod_name)
        if m:
            cand_parts.append((m + "." + entry).split("."))
    if entry.startswith("torch."):
        m = map_module(entry)
        if m:
            cand_parts.append(m.split("."))
    seen = set()
    for parts in cand_parts:
        key = ".".join(parts)
        if key in seen:
            continue
        seen.add(key)
        # try every split point: longest importable module prefix, rest as attrs
        for i in range(len(parts), 0, -1):
            try:
                base = importlib.import_module(".".join(parts[:i]))
            except Exception:
                continue
            obj = base
            for a in parts[i:]:
                obj = getattr(obj, a, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
    return None


def parse_page(path):
    """Return list of (level, title, [entries], [prose]) plus doc title.

    ``prose`` holds the narrative blocks (markdown paragraphs, code fences,
    eval-rst notes) that appear between the section heading and the
    autosummary table, so generated pages keep the upstream explanations
    instead of bare symbol listings. Directive fences (currentmodule /
    automodule / autosummary / autofunction / autoclass) are consumed; any
    other eval-rst fence is carried through verbatim.
    """
    text = open(path).read()
    lines = text.splitlines()
    sections = []  # (level, title, [(module, entry)], [prose lines])
    cur_mod = None
    doc_title = None
    i = 0
    in_fence = False
    in_eval = False
    eval_buf: list = []

    def add_prose(chunk):
        if sections and chunk:
            sections[-1][3].extend(chunk)

    def set_cur_mod(name):
        nonlocal cur_mod
        cur_mod = name

    def add_directives(body, current_section, set_module):
        """Consume sphinx directives inside one eval-rst fence."""
        pending = None  # (directive, name) from autofunction/autoclass lines
        in_summary = False

        def flush():
            nonlocal pending
            if pending:
                target = current_section()
                if target is not None:
                    target[2].append((cur_mod, pending))
                pending = None

        for l in body:
            m = re.match(r"^\s*\.\.\s+currentmodule::\s+(\S+)", l)
            if m:
                flush()
                in_summary = False
                set_module(m.group(1))
                continue
            m = re.match(r"^\s*\.\.\s+automodule::\s+(\S+)", l)
            if m:
                flush()
                in_summary = False
                set_module(m.group(1))
                continue
            m = re.match(r"^\s*\.\.\s+(autofunction|autoclass)::\s+([\w.]+)", l)
            if m:
                flush()
                in_summary = False
                pending = m.group(2)
                continue
            if re.match(r"^\s*\.\.\s+autosummary::", l):
                flush()
                in_summary = True
                continue
            if in_summary:
                if re.match(r"^\s*:[a-z_]+:", l):
                    continue
                name = l.strip()
                if not name:
                    continue
                if re.match(r"^[\w.]+$", name):
                    target = current_section()
                    if target is not None:
                        target[2].append((cur_mod, name))
                else:
                    in_summary = False
                continue
        flush()

    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            if not in_fence:
                in_fence = True
                in_eval = "eval-rst" in line
                eval_buf = [] if in_eval else None
                if not in_eval:
                    add_prose([line])
            else:
                in_fence = False
                if in_eval:
                    body = eval_buf
                    in_eval = False
                    add_directives(body, lambda: sections[-1] if sections else None,
                                   set_cur_mod)
                    joined = "\n".join(body)
                    has_directive = re.search(
                        r"^\s*\.\.\s+(?:currentmodule|automodule|autosummary"
                        r"|autofunction|autoclass)::", joined, re.M)
                    if not has_directive and body:
                        add_prose(["```{eval-rst}", *body, "```"])
                else:
                    add_prose([line])
            i += 1
            continue
        if in_fence:
            if in_eval:
                eval_buf.append(line)
            else:
                add_prose([line])
            i += 1
            continue
        m = re.match(r"^(#+)\s+(.*)$", line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            if doc_title is None and level == 1:
                doc_title = re.sub(r"[{}]", "", title)
            sections.append([level, title, [], []])
            i += 1
            continue
        m = re.match(r"^\s*\.\.\s+currentmodule::\s+(\S+)", line)
        if m:
            cur_mod = m.group(1)
            i += 1
            continue
        m = re.match(r"^\s*\.\.\s+automodule::\s+(\S+)", line)
        if m and cur_mod is None:
            cur_mod = m.group(1)
            i += 1
            continue
        m = re.match(r"^\s*\.\.\s+(?:autofunction|autoclass)::\s+([\w.]+)", line)
        if m:
            target = sections[-1] if sections else None
            if target is not None:
                target[2].append((cur_mod, m.group(1)))
            i += 1
            continue
        if re.match(r"^\s*\.\.\s+autosummary::", line):
            j = i + 1
            while j < len(lines) and re.match(r"^\s*:[a-z_]+:", lines[j]):
                j += 1
            entries = []
            while j < len(lines):
                l = lines[j]
                if not l.strip():
                    j += 1
                    continue
                if l.startswith("        ") or (l.startswith("    ") and not l.startswith("     ")):
                    name = l.strip()
                    if re.match(r"^[\w.]+$", name):
                        entries.append(name)
                    j += 1
                else:
                    break
            target = sections[-1] if sections else None
            if target is not None:
                for name in entries:
                    target[2].append((cur_mod, name))
            i = j
            continue
        if sections and line.strip():
            add_prose([line])
        i += 1
    return doc_title, sections


def public_additions(tp_mod_name, known_names):
    """Public symbols of a TensorPlay module not covered by source lists."""
    try:
        import importlib
        mod = importlib.import_module(tp_mod_name)
    except Exception:
        return []
    out = []
    for n in dir(mod):
        if n.startswith("_") or n in known_names or n.endswith("_backward"):
            continue
        if not n.isidentifier() or "pybind" in n.lower():
            continue
        o = getattr(mod, n)
        m = getattr(o, "__module__", "") or ""
        if not (m == "tensorplay" or m.startswith("tensorplay.")):
            continue
        if inspect.ismodule(o):
            continue
        if not (inspect.isclass(o) or callable(o)):
            continue
        # auto-generated wrappers whose only docstring is the C++ signature
        doc = getattr(o, "__doc__", None)
        if doc and "\n" not in doc.strip() and re.match(
                r"^[\w.]+\([^()]*\)\s*->", doc.strip()):
            continue
        out.append(n)
    return sorted(out)


def _importable(fq):
    import importlib
    parts = fq.split(".")
    for i in range(len(parts), 0, -1):
        try:
            base = importlib.import_module(".".join(parts[:i]))
        except Exception:
            continue
        obj = base
        for a in parts[i:]:
            obj = getattr(obj, a, None)
            if obj is None:
                return False
        return True
    return False


def emit_name(mod_name, entry):
    """Fully-qualified tensorplay name to document for a resolved symbol."""
    short = entry.split(".")[-1].split("(")[0]
    obj = resolve(mod_name, entry)
    candidates = []
    if obj is not None:
        m = getattr(obj, "__module__", "") or ""
        if m == "tensorplay" or m.startswith("tensorplay."):
            candidates.append(m + "." + short)
    mm = map_module(mod_name) if mod_name else None
    candidates.append((mm or "tensorplay") + "." + short)
    for fq in candidates:
        # __module__ can lie (C-ext re-exports); only emit names sphinx will
        # be able to import when generating the stub.
        if _importable(fq):
            return fq
    return None


def scrub_prose(text):
    """Brand + link scrubbing for carried-over upstream prose."""
    text = (text
            .replace("torchvision", "tensorplay.vision")
            .replace("torchaudio", "tensorplay.audio")
            .replace("TORCH_DOCTEST_", "TP_DOCTEST_")
            .replace("TorchInductor", "the TP compiler"))
    # RST roles referencing torch symbols -> tensorplay equivalents.
    text = re.sub(r"(:\w+:)`([^`]*?)\btorch\.", r"\1`\2tensorplay.", text)
    text = re.sub(r"\btorch\.", "tensorplay.", text)
    text = re.sub(r"\btorch\b", "tensorplay", text)
    text = text.replace("PyTorch", "TensorPlay").replace("pytorch", "tensorplay")
    # URL normalization last: word scrubbing above rewrites upstream host
    # names (e.g. docs.pytorch.org -> docs.tensorplay.org) that then need to
    # point at the project site with the right path structure.
    return (text
            .replace("https://github.com/lexing-2026/TensorPlay/ao", "https://github.com/lexing-2026/TensorPlay")
            .replace("https://www.tensorplay.cn/docs/docs/", "https://www.tensorplay.cn/docs/")
            .replace("https://docs.tensorplay.org/", "https://www.tensorplay.cn/docs/")
            .replace("docs.tensorplay.org", "www.tensorplay.cn")
            .replace("https://www.tensorplay.cn/docs/tutorials", "https://www.tensorplay.cn/docs")
            .replace("https://github.com/pytorch/", "https://github.com/lexing-2026/TensorPlay/"))


def strip_upstream_only_blocks(text):
    """Remove upstream-lifecycle announcements that do not apply to TP.

    torch.ao quantization was deprecated in favor of the external torchao
    project; tensorplay keeps its own tensorplay.quantization package, so the
    migration notice would mislead users.
    """
    return re.sub(
        r"We are centralizing all quantization[\s\S]*?cleared\.\n\n",
        "Quantization is provided by the ``tensorplay.quantization`` package.\n\n",
        text)


def render(page_title, sections, extras=None):
    buf = []
    render._prev_level = 1
    page_seen = set()
    page_seen_lower = set()
    buf.append(HIDDEN_ROLE)
    buf.append(f"# {page_title}\n")
    known = set()
    resolved_total = dropped = 0
    rendered = []  # (level, title, [(mod, entry, ok)], prose)
    for level, title, entries, prose in sections:
        row = []
        for mod, name in entries:
            obj = resolve(mod, name)
            known.add(name.split(".")[-1])
            if obj is None:
                dropped += 1
                row.append((mod, name, False))
            else:
                resolved_total += 1
                row.append((mod, name, True))
        rendered.append((level, title, row, prose))
    # Manual injections for APIs whose source scoping does not map cleanly.
    if extras:
        extra_by_title = {}
        for sec_title, names in extras.items():
            mods = []
            pairs = []
            for mod, name in names:
                if resolve(mod, name) is not None:
                    pairs.append((mod, name))
                    mods.append(map_module(mod))
                    known.add(name.split(".")[-1])
                    resolved_total += 1
            if pairs:
                extra_by_title[sec_title] = (mods, pairs)
        for level, title, row, prose in rendered:
            if title in extra_by_title:
                mods, pairs = extra_by_title.pop(title)
                row.extend((m, n, True) for m, n in pairs)
                row_mods = {map_module(e[0]) or e[0] for e in row if e[2]}
                # currentmodule emission handled below via union
                pass
        # Any extras whose section did not appear in the source go at the end.
        for sec_title, (mods, pairs) in extra_by_title.items():
            rendered.append((2, sec_title, [(m, n, True) for m, n in pairs], []))
    for level, title, row, prose in rendered:
        ok = [e for e in row if e[2]]
        # Narrative prose is carried even for sections whose symbols were
        # dropped wholesale — the text still documents the topic.
        if not ok and not prose:
            continue
        # clamp heading levels so the document never jumps (H1 -> H3), which
        # MyST reports this as a warning; source pages occasionally skip levels.
        prev_level = getattr(render, "_prev_level", 1)
        level = min(level, prev_level + 1)
        render._prev_level = level
        # The page-level h1 already renders the document title; upstream
        # source pages duplicate the heading around their automodule block,
        # which produced stacked identical headings. Merge instead: keep the
        # prose and entries, drop only the redundant heading line.
        is_page_h1 = level == 1 and scrub_prose(title) == page_title
        if not is_page_h1:
            buf.append(f"{'#' * level} {scrub_prose(title)}\n")
        if prose:
            body = scrub_prose("\n".join(prose)).strip("\n")
            if body:
                buf.append(body + "\n")
        # Fully-qualified names anchored at each symbol's real home module:
        # namespace does not re-export everything, so short names under a
        # page-level currentmodule break sphinx's stub imports.
        seen_names = page_seen
        seen_lower = page_seen_lower
        entry_lines = []
        for e in ok:
            fq = emit_name(e[0], e[1])
            if fq is None or fq in seen_names:
                continue
            # case-insensitive dedupe across the whole page: autosectionlabel
            # treats Device/device stubs as colliding labels
            if fq.lower() in seen_lower:
                continue
            seen_names.add(fq)
            seen_lower.add(fq.lower())
            entry_lines.append("    " + fq)
        if entry_lines:
            buf.append("```{eval-rst}\n.. autosummary::\n    :toctree: generated"
                       "\n    :nosignatures:\n\n" + "\n".join(entry_lines) + "\n```\n")
    return "\n".join(buf), resolved_total, dropped, known


PAGES = [
    ("torch.md", "tensorplay.md"),
    ("autograd.md", "autograd.md"),
    ("nn.md", "nn.md"),
    ("nn.functional.md", "nn.functional.md"),
    ("nn.init.md", "nn.init.md"),
    ("optim.md", "optim.md"),
    ("cuda.md", "cuda.md"),
    ("amp.md", "amp.md"),
    ("linalg.md", "linalg.md"),
    ("fft.md", "fft.md"),
    ("special.md", "special.md"),
    ("sparse.md", "sparse.md"),
    ("random.md", "random.md"),
    ("data.md", "data.md"),
    ("checkpoint.md", "checkpoint.md"),
    ("futures.md", "futures.md"),
    ("hub.md", "hub.md"),
    ("multiprocessing.md", "multiprocessing.md"),
    ("library.md", "library.md"),
    ("distributed.md", "distributed.md"),
    ("quantization.md", "quantization.md"),
]

EXTRAS = {
    "amp.md": {
        "Autocasting": [("torch.amp", "autocast")],
        "Gradient Scaling": [("torch.amp", "GradScaler"), ("torch.amp", "custom_fwd"),
                              ("torch.amp", "custom_bwd"),
                              ("torch.amp", "is_autocast_available")],
    },
    "random.md": {
        "Random Generator": [
            ("torch.random", "Generator"), ("torch.random", "manual_seed"),
            ("torch.random", "seed"), ("torch.random", "initial_seed"),
            ("torch.random", "get_rng_state"), ("torch.random", "set_rng_state"),
            ("torch.random", "fork_rng"), ("torch.random", "default_generator"),
        ],
    },
    "multiprocessing.md": {
        "API Reference": [
            ("torch.multiprocessing", "reduce_tensor"),
            ("torch.multiprocessing", "allow_connection_pickling"),
            ("torch.multiprocessing", "set_start_method"),
            ("torch.multiprocessing", "get_start_method"),
            ("torch.multiprocessing", "get_all_start_methods"),
        ],
    },
}

# Pages that get an explicit trailing section listing public tensorplay-only
# Symbols not covered by the source lists. Value = (TensorPlay module, exclude set).
ADDITIONS = {
    "tensorplay.md": ("tensorplay", {"autocast_decrement_nesting", "autocast_increment_nesting"}),
    "nn.md": ("tensorplay.nn", set()),
    "nn.functional.md": ("tensorplay.nn.functional", set()),
    "optim.md": ("tensorplay.optim", set()),
    "cuda.md": ("tensorplay.cuda", {"cudaStatus"}),
}

def main():
    for src, dst in PAGES:
        path = f"{REFERENCE_SOURCE}/{src}"
        try:
            title, sections = parse_page(path)
        except FileNotFoundError:
            print(f"skip {src}: source page missing")
            continue
        if title is None:
            title = dst.replace(".md", "")
        title = scrub_prose(title)
        if title == "torch":
            title = "tensorplay"
        body, n_ok, n_drop, known = render(title, sections, extras=EXTRAS.get(dst))
        body = strip_upstream_only_blocks(body)
        if dst in ADDITIONS:
            tp_mod, excl = ADDITIONS[dst]
            extra = [n for n in public_additions(tp_mod, known) if n not in excl]
            if extra:
                mod_line = f"```{{eval-rst}}\n.. currentmodule:: {tp_mod}\n```\n"
                lines = "\n".join("    " + n for n in extra)
                block = ("\n## TensorPlay-specific additions\n\n" + mod_line +
                         "```{eval-rst}\n.. autosummary::\n    :toctree: generated"
                         "\n    :nosignatures:\n\n" + lines + "\n```\n")
                body += block
                n_ok += len(extra)
        open(f"{OUT}/{dst}", "w").write(body + "\n")
        print(f"{dst}: {n_ok} resolved, {n_drop} dropped (missing in tensorplay)")

    # Narrative developer notes: copy verbatim with brand scrubbing. These
    # pages carry the upstream explanations (broadcasting rules, pickling
    # format, determinism, module mechanics) that the reference pages only
    # summarize. Pure prose — no symbol resolution needed.
    notes_out = os.path.join(OUT, "notes")
    os.makedirs(notes_out, exist_ok=True)
    for name in NOTES_PAGES:
        src_path = os.path.join(REFERENCE_SOURCE, "notes", f"{name}.md")
        if not os.path.exists(src_path):
            print(f"skip notes/{name}.md: source page missing")
            continue
        text = open(src_path).read()
        # Drop trailing autofunction fences: those APIs live on reference
        # pages already; keep everything else verbatim.
        text = re.sub(
            r"```{eval-rst}\n(?:.. autofunction:: [\w.]+\n?)+```",
            "", text)
        open(os.path.join(notes_out, f"{name}.md"), "w").write(
            strip_upstream_only_blocks(scrub_prose(text)) + "\n")
    # Index page for the notes section, mirroring upstream's notes.md.
    open(os.path.join(notes_out, "index.md"), "w").write(
        "(developer-notes)=\n# Developer Notes\n\n"
        "```{toctree}\n:glob:\n:maxdepth: 1\n\nnotes/*\n```\n")


NOTES_PAGES = [
    "broadcasting",
    "serialization",
    "randomness",
    "modules",
    "faq",
    "numerical_accuracy",
    "autograd",
    "amp_examples",
    "gradcheck",
    "out",
]


if __name__ == "__main__":
    main()
