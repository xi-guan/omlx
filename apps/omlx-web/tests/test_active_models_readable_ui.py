# SPDX-License-Identifier: Apache-2.0
"""Active Models keeps one line per model on wide screens; details live in a popover."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "omlx_web"


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_active_models_rows_keep_one_height():
    template = _read("templates/dashboard/blocks/_active_models.html")
    dashboard = _read("templates/dashboard.html")
    css = _read("static/css/dashboard.css")

    assert dashboard.index("js/active_models.js") < dashboard.index("js/dashboard.js")
    assert 'class="hidden lg:flex items-center justify-between gap-3 h-6"' in template
    assert 'tabindex="0" role="group"' in template
    assert ':aria-describedby="ActiveModels.popoverId(m.id)"' in template
    assert '<div class="am-detail" :id="ActiveModels.popoverId(m.id)"' in template
    # the totals footer stays rendered at zero, so a request starting never resizes the card
    assert 'x-show="stats.active_models.total_active_requests > 0 ||' not in template
    assert ".am-detail.am-detail-open { display: block; }" in css


def test_usage_table_reuses_the_model_name_markup():
    usage = _read("templates/dashboard/_usage.html")
    active = _read("templates/dashboard/blocks/_active_models.html")

    for template in (usage, active):
        assert "{% include 'dashboard/_model_logo.html' %}" in template
        assert "{% include 'dashboard/_model_name_detail.html' %}" in template
    assert "ActiveModels.describe(row.model_id)" in usage
    assert ":aria-describedby=\"'usage-' + ActiveModels.popoverId(row.model_id)\"" in usage
    # the table scrolls inside overflow containers, so its popover must leave them
    assert '<template x-teleport="body">' in usage
    assert "ActiveModels.dashboardModels()" in usage
    # the model filter keeps listing raw ids
    assert '<option :value="id" x-text="id"></option>' in usage


def test_active_models_new_keys_exist_in_every_locale():
    used = set(
        re.findall(
            r"status\.active_models\.[a-z_]+(?:\.[a-z_]+)?",
            _read("templates/dashboard/blocks/_active_models.html")
            + _read("templates/dashboard/_model_name_detail.html")
            + _read("static/js/active_models.js"),
        )
    )
    en = json.loads(_read("i18n/en.json"))
    terms = {k for k in en if k.startswith("status.active_models.term.")}
    # the bare prefix comes from the TERM constant in active_models.js
    keys = (used | terms) - {"status.active_models.term"}
    assert len(terms) >= 25
    zh = json.loads(_read("i18n/zh.json"))
    for path in sorted((ROOT / "i18n").glob("*.json")):
        missing = keys - json.loads(path.read_text(encoding="utf-8")).keys()
        assert not missing, f"{path.name} is missing {sorted(missing)}"
    # zh is the user's language: every new explanation carries its own translation
    untranslated = sorted(k for k in terms if zh[k] == en[k])
    assert not untranslated, untranslated
