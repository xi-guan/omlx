# SPDX-License-Identifier: Apache-2.0
"""Static guard that dashboard.js reads only state it declares.

Upstream rebases keep reintroducing markup whose Alpine state this fork has
removed (and vice versa), which renders a tab blank without failing any test:
pytest never evaluates the template/JS binding. This check does.
"""

import re
from pathlib import Path

# Alpine injects these into the component at runtime.
_ALPINE_MAGICS = {"$nextTick", "$refs", "$watch", "$el", "$dispatch", "$store"}

# Assigned lazily and never rendered, so they need no reactive declaration.
_UNDECLARED_OK = {"_accPollTimer", "uploadRedownloadNotice"}

# Top-level members of the object literal returned by dashboard().
_DECL_RE = re.compile(
    r"^            (?:async\s+|get\s+|set\s+)?([A-Za-z_$][\w$]*)\s*[:(]", re.M
)
_USE_RE = re.compile(r"this\.([A-Za-z_$][\w$]*)")


def _component_body() -> str:
    root = Path(__file__).resolve().parents[1]
    lines = (root / "omlx/admin/static/js/dashboard.js").read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "return {")
    end = next(i for i, ln in enumerate(lines) if ln == "    }")
    return "\n".join(lines[start:end])


def test_dashboard_reads_only_declared_state():
    body = _component_body()
    declared = set(_DECL_RE.findall(body))
    used = set(_USE_RE.findall(body))

    missing = sorted(used - declared - _ALPINE_MAGICS - _UNDECLARED_OK)
    assert not missing, (
        f"dashboard.js reads undeclared state: {missing}. Either declare it in "
        "the dashboard() object literal or delete the dead reader."
    )


def test_dashboard_carries_no_modelscope_state():
    body = _component_body()
    leftovers = sorted(
        {n for n in _USE_RE.findall(body) if re.match(r"_?ms[A-Z]", n)}
        | set(re.findall(r"[A-Za-z_$][\w$]*[Mm][Ss][A-Z][\w$]*\s*[:(]", body))
    )
    assert not leftovers, f"fork is HuggingFace-only; ModelScope state is back: {leftovers}"
