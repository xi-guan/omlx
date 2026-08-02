# SPDX-License-Identifier: Apache-2.0
"""Inverse text normalization (ITN) for Chinese ASR output.

Qwen3-ASR (and other plain transcription models) emit numbers in spoken form
(一百二十三). The cloud Qwen3-ASR service applies ITN server-side via an
`enable_itn` flag; the mlx-audio local port does not, so we reproduce that step
here. cn2an handles the conversion; a protect list shields spans that look
numeric but must stay verbatim (第一, 十分, 星期一, ...).
"""

import re

# Spans that contain number characters but are not quantities. Matched and
# masked before cn2an runs, then restored unchanged. Order matters: longer /
# more specific patterns should appear before the bare characters they contain.
_NUM = "零一二三四五六七八九十百千万亿两"

_PROTECT_PATTERNS = [
    r"第[一二三四五六七八九十百千零两]+",  # ordinals: 第一, 第二十
    r"十分(?![一二三四五六七八九十之])",  # 十分(感谢); keep 十分钟/十分之一 convertible
    r"星期[一二三四五六日天]", r"周[一二三四五六日天]",
    r"初[一二三四五六七八九十]",  # lunar dates: 初一
    # adjacent-digit pairs are an approximation ("two or three weeks"), not a
    # two-digit number; the trailing 十 keeps 三四十 whole
    r"(?:一两|两三|三四|三五|四五|五六|五七|六七|七八|八九)十?",
    r"十几", r"几十", r"一一",
    # a lone 一 is the article "a/an" or a reduplication marker, never a written
    # digit: 另外一个, 保持一致, 想一想, 唯一. Adjacency to another numeral means
    # it really is one (一百二十三, 十一, 电话是一三八), and 分之一 stays a fraction
    rf"(?<![{_NUM}点])(?<!分之)一(?![{_NUM}]|点[{_NUM}])",
    # a lone 两 is the measure-word "two of": 两个, 等两天, 说两句. It is a real
    # digit only in front of a magnitude: 两千零五年, 两百三十
    rf"(?<![{_NUM}])两(?![百千万亿])",
]

_PROTECT_RE = [re.compile(p) for p in _PROTECT_PATTERNS]
_SENTINEL = "\x00"


def itn(text: str) -> str:
    """Convert spoken Chinese numerals to arabic digits, protecting idioms.

    Returns the input unchanged on any cn2an failure.
    """
    if not text:
        return text

    try:
        import cn2an
    except ImportError:
        return text

    holes: list[str] = []

    def _stash(m: "re.Match[str]") -> str:
        holes.append(m.group(0))
        return f"{_SENTINEL}{len(holes) - 1}{_SENTINEL}"

    masked = text
    for pat in _PROTECT_RE:
        masked = pat.sub(_stash, masked)

    try:
        masked = cn2an.transform(masked, "cn2an")
    except Exception:
        return text

    for i, original in enumerate(holes):
        masked = masked.replace(f"{_SENTINEL}{i}{_SENTINEL}", original)
    return masked
