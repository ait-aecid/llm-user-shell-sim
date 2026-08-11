from __future__ import annotations

"""Audit-log exploratory feature extraction for AI-vs-human comparison.

This module parses Linux auditd ``audit.log`` files into structured per-event and
per-command records, then derives families of behavioral features (volume,
timing, command vocabulary, syscall fingerprint, sequence, session/process-tree,
MITRE-key composition). The features are intended for *visual* EDA: each family
function returns both a per-actor summary and pooled per-event/per-command arrays
that downstream plotting code renders by class (AI vs human).

Design notes
------------
* Audit records that belong to the same event share an id ``audit(ts:serial)``.
  We group the multi-line record types (SYSCALL/EXECVE/PATH/SOCKADDR/PROCTITLE/...)
  by that id into a single :class:`AuditEvent`.
* ``PROCTITLE`` and ``EXECVE`` carry the executed command line, hex-encoded when it
  contains NUL separators or non-printable bytes. We decode these into real command
  strings so flags, pipes, redirects and chaining are recoverable.
* Parsing is permissive: malformed/missing fields become ``None`` and are simply
  excluded from the metric that needs them.
"""

import gzip
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from src.core.shared.actor_catalog import is_ai_actor
from src.core.stats.data_catalog import analysis_actors, get_log_path

# ---- Regexes ----

# audit(<epoch.frac>:<serial>)
AUDIT_ID_RE = re.compile(r"msg=audit\((?P<ts>\d+(?:\.\d+)?):(?P<serial>\d+)\)")
TYPE_RE = re.compile(r"^type=(?P<type>\S+)")
# key=value where value is a quoted string or a bare token.
FIELD_RE_TEMPLATE = r'{key}=(?P<val>"[^"]*"|\S+)'

# Interactive editors/pagers/monitors hypothesized to be human-dominated.
INTERACTIVE_TOOLS = (
    "vim", "vi", "nano", "emacs", "less", "more", "man", "top", "htop", "nvim",
)
# File-operation commands for the file-op mix.
FILE_OP_COMMS = ("cp", "rm", "mv", "mkdir", "rmdir", "touch", "chmod", "chown", "ln")
# Tokens that indicate command chaining / composition in a reconstructed line.
CHAIN_TOKENS = ("|", "&&", "||", ";", ">", ">>", "<")


def _field(line: str, key: str) -> Optional[str]:
    """Return the (unquoted) value of ``key`` in an audit line, or ``None``."""
    m = re.search(FIELD_RE_TEMPLATE.format(key=re.escape(key)), line)
    if not m:
        return None
    return m.group("val").strip('"')


def _maybe_hex_decode(token: str) -> str:
    """Decode an audit hex-encoded token to text, leaving plain tokens untouched.

    Auditd hex-encodes arguments that contain spaces/NULs/non-printables. Such a
    token is an even-length run of hex digits; we decode it, turning embedded NUL
    separators into spaces. Anything that is not clean hex is returned unchanged.
    """
    if token is None:
        return ""
    t = token.strip('"')
    if len(t) >= 2 and len(t) % 2 == 0 and re.fullmatch(r"[0-9A-Fa-f]+", t):
        try:
            raw = bytes.fromhex(t)
            return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        except ValueError:
            return t
    return t


@dataclass
class AuditEvent:
    """One audit event: all record lines sharing an ``audit(ts:serial)`` id."""

    ts: float
    serial: str
    types: set[str] = field(default_factory=set)
    syscall: Optional[str] = None
    success: Optional[str] = None
    exit: Optional[str] = None
    comm: Optional[str] = None
    exe: Optional[str] = None
    tty: Optional[str] = None
    ses: Optional[str] = None
    pid: Optional[str] = None
    ppid: Optional[str] = None
    uid: Optional[str] = None
    euid: Optional[str] = None
    key: Optional[str] = None
    cmdline: Optional[str] = None  # reconstructed command line (EXECVE preferred)


def _decode_execve(line: str) -> Optional[str]:
    """Reconstruct a command line from an EXECVE record's a0..aN args."""
    args = re.findall(r'a\d+="?([^"\s]*)"?', line)
    if not args:
        return None
    return " ".join(_maybe_hex_decode(a) for a in args).strip() or None


def parse_audit_file(path: str | Path) -> list[AuditEvent]:
    """Parse an ``audit.log`` into a list of :class:`AuditEvent`, time-ordered.

    Records are grouped by their ``audit(ts:serial)`` id. SYSCALL fields populate
    the core attributes; EXECVE/PROCTITLE populate the reconstructed command line
    (EXECVE preferred, PROCTITLE as fallback).
    """
    events: dict[str, AuditEvent] = {}
    order: list[str] = []
    path = Path(path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            idm = AUDIT_ID_RE.search(line)
            tym = TYPE_RE.search(line)
            if not idm or not tym:
                continue
            rtype = tym.group("type")
            key = f"{idm.group('ts')}:{idm.group('serial')}"
            ev = events.get(key)
            if ev is None:
                ev = AuditEvent(ts=float(idm.group("ts")), serial=idm.group("serial"))
                events[key] = ev
                order.append(key)
            ev.types.add(rtype)

            if rtype == "SYSCALL":
                ev.syscall = _field(line, "SYSCALL") or _field(line, "syscall")
                ev.success = _field(line, "success")
                ev.exit = _field(line, "exit")
                ev.comm = _field(line, "comm")
                ev.exe = _field(line, "exe")
                ev.tty = _field(line, "tty")
                ev.ses = _field(line, "ses")
                ev.pid = _field(line, "pid")
                ev.ppid = _field(line, "ppid")
                ev.uid = _field(line, "UID") or _field(line, "uid")
                ev.euid = _field(line, "EUID") or _field(line, "euid")
                ev.key = _field(line, "key")
            elif rtype == "EXECVE":
                cmd = _decode_execve(line)
                if cmd:
                    ev.cmdline = cmd
            elif rtype == "PROCTITLE" and ev.cmdline is None:
                pt = _field(line, "proctitle")
                if pt:
                    ev.cmdline = _maybe_hex_decode(pt)

    return [events[k] for k in order]


# ---- generic helpers ----

def _entropy(counts: Iterable[int]) -> float:
    """Shannon entropy (bits) of a count distribution; 0 for empty/degenerate."""
    counts = [c for c in counts if c > 0]
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts)


def _command_stream(events: list[AuditEvent]) -> list[str]:
    """Ordered list of executed command names (argv[0] basename) for EXECVE events."""
    stream: list[str] = []
    for ev in events:
        if "EXECVE" in ev.types and ev.cmdline:
            argv0 = ev.cmdline.split(" ", 1)[0]
            stream.append(Path(argv0).name or argv0)
    return stream


# ---- Family A: volume, rate & composition ----

def family_volume(events: list[AuditEvent]) -> dict:
    """Per-actor volume/rate/composition summary."""
    n = len(events)
    ts = [e.ts for e in events]
    span_min = (max(ts) - min(ts)) / 60.0 if len(ts) >= 2 else 0.0
    tcount = Counter(t for e in events for t in e.types)
    syscalls = tcount.get("SYSCALL", 0) or 1
    return {
        "n_events": n,
        "span_min": round(span_min, 2),
        "events_per_min": round(n / span_min, 2) if span_min > 0 else 0.0,
        "ratio_path_syscall": round(tcount.get("PATH", 0) / syscalls, 3),
        "ratio_execve_syscall": round(tcount.get("EXECVE", 0) / syscalls, 3),
        "ratio_sockaddr_syscall": round(tcount.get("SOCKADDR", 0) / syscalls, 3),
    }


# ---- Family B: timing & rhythm ----

def family_timing(events: list[AuditEvent]) -> dict:
    """Inter-event/inter-command timing summary plus pooled delay arrays."""
    ts = sorted(e.ts for e in events)
    gaps = [b - a for a, b in zip(ts, ts[1:]) if b >= a]
    cmd_ts = sorted(e.ts for e in events if "EXECVE" in e.types)
    cmd_gaps = [b - a for a, b in zip(cmd_ts, cmd_ts[1:]) if b >= a]

    def _median(xs: list[float]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

    sub100 = sum(1 for g in gaps if g < 0.1) / len(gaps) if gaps else 0.0
    long_pauses = sum(1 for g in cmd_gaps if g > 30.0)
    return {
        "median_event_gap_s": round(_median(gaps), 4),
        "frac_sub100ms_gaps": round(sub100, 4),
        "median_cmd_gap_s": round(_median(cmd_gaps), 3),
        "long_pause_count_30s": long_pauses,
        "_pool_event_gaps": gaps,
        "_pool_cmd_gaps": cmd_gaps,
    }


# ---- Family C: command vocabulary & process behavior ----

def family_commands(events: list[AuditEvent]) -> dict:
    """Command-vocabulary, interactive-tool, flag and chaining summary."""
    stream = _command_stream(events)
    counts = Counter(stream)
    distinct = len(counts)
    total = sum(counts.values()) or 1
    ttr = distinct / total

    interactive = sum(counts.get(t, 0) for t in INTERACTIVE_TOOLS) / total
    repeats = sum(c - 1 for c in counts.values() if c > 1) / total

    argcs: list[int] = []
    flags_per_cmd: list[int] = []
    chained = 0
    n_cmd = 0
    for ev in events:
        if "EXECVE" in ev.types and ev.cmdline:
            toks = ev.cmdline.split()
            n_cmd += 1
            argcs.append(len(toks))
            flags_per_cmd.append(sum(1 for t in toks[1:] if t.startswith("-")))
            if any(ct in toks for ct in CHAIN_TOKENS):
                chained += 1
    return {
        "distinct_commands": distinct,
        "command_ttr": round(ttr, 4),
        "interactive_tool_rate": round(interactive, 4),
        "repeat_command_rate": round(repeats, 4),
        "mean_argc": round(sum(argcs) / len(argcs), 2) if argcs else 0.0,
        "mean_flags_per_cmd": round(sum(flags_per_cmd) / len(flags_per_cmd), 3) if flags_per_cmd else 0.0,
        "chained_command_rate": round(chained / n_cmd, 4) if n_cmd else 0.0,
        "_pool_argc": argcs,
        "_tool_counts": {t: counts.get(t, 0) for t in INTERACTIVE_TOOLS},
        "_cmd_total": total,
    }


# ---- Family D: syscall-level fingerprint ----

def family_syscall(events: list[AuditEvent]) -> dict:
    """Syscall distribution, entropy, failure and file-op/privilege profile."""
    sysc = Counter(e.syscall for e in events if e.syscall)
    total_sys = sum(sysc.values()) or 1
    succ = Counter(e.success for e in events if e.success)
    fail_rate = succ.get("no", 0) / (succ.get("yes", 0) + succ.get("no", 0) or 1)

    comms = Counter(e.comm for e in events if e.comm)
    fileop = sum(comms.get(c, 0) for c in FILE_OP_COMMS)
    priv = sum(v for k, v in sysc.items() if k in ("setuid", "setgid", "setresuid", "setresgid", "setreuid", "setregid"))
    return {
        "syscall_entropy_bits": round(_entropy(sysc.values()), 3),
        "distinct_syscalls": len(sysc),
        "fail_rate": round(fail_rate, 4),
        "fileop_comm_rate": round(fileop / (sum(comms.values()) or 1), 4),
        "privilege_syscall_rate": round(priv / total_sys, 4),
        "_syscall_share": {k: v / total_sys for k, v in sysc.items()},
    }


# ---- Family E: sequence / ordering ----

def family_sequence(events: list[AuditEvent]) -> dict:
    """Command bigram entropy, novel-bigram rate and stream compressibility."""
    stream = _command_stream(events)
    bigrams = Counter(zip(stream, stream[1:]))
    nb = sum(bigrams.values()) or 1
    novel = sum(1 for c in bigrams.values() if c == 1) / nb

    joined = "\n".join(stream).encode("utf-8")
    if joined:
        ratio = len(gzip.compress(joined, 9)) / len(joined)
    else:
        ratio = 0.0
    return {
        "command_bigram_entropy_bits": round(_entropy(bigrams.values()), 3),
        "novel_bigram_rate": round(novel, 4),
        "command_gzip_ratio": round(ratio, 4),
    }


# ---- Family F: session / process-tree ----

def family_session(events: list[AuditEvent]) -> dict:
    """tty interactivity, distinct sessions and process-tree branching summary."""
    tty = Counter(e.tty for e in events if e.tty)
    interactive_tty = sum(v for k, v in tty.items() if k and k != "(none)")
    distinct_ses = len({e.ses for e in events if e.ses and e.ses != "4294967295"})

    children: dict[str, set[str]] = {}
    for e in events:
        if e.ppid and e.pid:
            children.setdefault(e.ppid, set()).add(e.pid)
    branching = (sum(len(c) for c in children.values()) / len(children)) if children else 0.0
    return {
        "interactive_tty_rate": round(interactive_tty / (sum(tty.values()) or 1), 4),
        "distinct_sessions": distinct_ses,
        "mean_proc_branching": round(branching, 3),
    }


# ---- Family G: MITRE technique-key composition ----

def family_mitre(events: list[AuditEvent]) -> dict:
    """Distribution over injected ATT&CK ``key=`` tags."""
    keys = Counter(e.key for e in events if e.key)
    total = sum(keys.values()) or 1
    return {
        "distinct_mitre_keys": len(keys),
        "_key_share": {k: v / total for k, v in keys.items()},
    }


# ---- Family H: complexity indices (template cluster-id distributions) ----

# Window/stride for the sequence variants: the "medium-scale behavior" setting
# from the complexity_metrics_runner sweep.
COMPLEXITY_WINDOW = 10
COMPLEXITY_STRIDE = 2


def _add_complexity(feats: list["ActorFeatures"], dataset: str) -> None:
    """Attach complexity indices to each actor, computed in a shared template space.

    All actors' preprocessed audit lines are Drain-mined with ONE miner so the
    cluster-id vocabulary is comparable across actors; the cid stream is then
    split back per actor. Cannot run inside per-actor extraction for this reason.
    """
    # Lazy imports: pulls drain3/scipy only when extraction actually runs.
    from src.core.shared.loader import (
        LoadConfig,
        _assign_templates_and_cids_global,
        _create_template_miner,
        _get_drain_ini,
    )
    from src.stats_tools.complexity_metrics import (
        load_preprocessed_lines,
        stats_from_ids,
        stats_from_windows,
    )

    per_actor = [
        load_preprocessed_lines(
            get_log_path(f.actor, "audit", dataset=dataset), preprocess_mode="soft"
        )
        for f in feats
    ]
    miner = _create_template_miner(ini_path=_get_drain_ini(LoadConfig()))
    _, cids = _assign_templates_and_cids_global(
        miner, [line for lines in per_actor for line in lines]
    )
    pos = 0
    for f, lines in zip(feats, per_actor):
        actor_cids = cids[pos:pos + len(lines)]
        pos += len(lines)
        metrics = stats_from_ids(actor_cids)
        metrics.update(stats_from_windows(actor_cids, COMPLEXITY_WINDOW, COMPLEXITY_STRIDE))
        f.complexity = {f"complexity_{k}": round(v, 4) for k, v in metrics.items()}


# ---- top-level driver ----

@dataclass
class ActorFeatures:
    """All extracted features for one actor."""

    actor: str
    dataset: str
    is_ai: bool
    volume: dict
    timing: dict
    commands: dict
    syscall: dict
    sequence: dict
    session: dict
    mitre: dict
    complexity: dict = field(default_factory=dict)


def extract_actor(actor: str, dataset: str) -> Optional[ActorFeatures]:
    """Parse one actor's audit log and compute every feature family.

    Returns ``None`` (with no exception) if the log is missing or has no events,
    so a dataset-wide run never crashes on a single bad actor.
    """
    path = get_log_path(actor, "audit", dataset=dataset)
    events = parse_audit_file(path)
    if not events:
        return None
    return ActorFeatures(
        actor=actor,
        dataset=dataset,
        is_ai=is_ai_actor(actor),
        volume=family_volume(events),
        timing=family_timing(events),
        commands=family_commands(events),
        syscall=family_syscall(events),
        sequence=family_sequence(events),
        session=family_session(events),
        mitre=family_mitre(events),
    )


def extract_dataset(dataset: str) -> list[ActorFeatures]:
    """Extract features for every analysis actor in a dataset (skipping empties)."""
    out: list[ActorFeatures] = []
    for actor in analysis_actors(dataset):
        feats = extract_actor(actor, dataset)
        if feats is not None:
            out.append(feats)
    if out:
        _add_complexity(out, dataset)
    return out
