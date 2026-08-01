# SPDX-License-Identifier: Apache-2.0
"""Tests for Chinese ITN (inverse text normalization).

Qwen3-ASR emits chinese numerals; omlx.engine.itn.itn converts them to arabic
digits (matching the cloud service's server-side ITN) while a protect list
shields idioms that look numeric but must stay verbatim (第一, 十分, 星期一).
"""

import pytest

from omlx.engine.itn import itn

# itn() returns its input unchanged when cn2an is missing, which would silently
# turn the protect-list cases below into vacuous passes. cn2an ships in the
# [audio] extra, so skip rather than assert against the degraded no-op path.
pytest.importorskip("cn2an")


@pytest.mark.parametrize(
    "src,want",
    [
        # numbers that should convert
        ("我有一百二十三块钱，电话是一三八", "我有123块钱，电话是138"),
        ("我等了十三点五分钟", "我等了13.5分钟"),
        ("两千零五年百分之三十", "2005年30%"),
        # protected idioms — must stay verbatim
        ("第一次见面", "第一次见面"),
        ("十分感谢你", "十分感谢你"),
        ("一下子就好了", "一下子就好了"),
        ("星期一开会", "星期一开会"),
        # mixed: protect 第一, still convert 九十八
        ("他考了第一名得了九十八分", "他考了第一名得了98分"),
        ("一般来说三个人", "一般来说3个人"),
        # a lone 一 is the article / reduplication marker, not a digit
        ("另外一个方式", "另外一个方式"),
        ("你再想一想", "你再想一想"),
        ("保持一致", "保持一致"),
        ("这是唯一的统一入口", "这是唯一的统一入口"),
        ("一一对应", "一一对应"),
        ("其中之一", "其中之一"),
        # ... but a 一 next to another numeral really is one
        ("电话是一三八", "电话是138"),
        ("二十一个人", "21个人"),
        ("十分之一", "1/10"),
        ("十一月", "11月"),
        # adjacent-digit pairs are approximations, not two-digit numbers
        ("距离上次两三周了", "距离上次两三周了"),
        ("三四天以后", "三四天以后"),
        ("三四十个人", "三四十个人"),
        ("一两个问题", "一两个问题"),
        ("十几个和几十个", "十几个和几十个"),
    ],
)
def test_itn_conversions_and_protections(src, want):
    assert itn(src) == want


def test_itn_empty_input():
    assert itn("") == ""


def test_itn_no_numbers_unchanged():
    assert itn("今天天气不错") == "今天天气不错"
