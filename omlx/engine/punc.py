# SPDX-License-Identifier: Apache-2.0
"""Punctuation restoration for ASR output that has none.

FireRedASR2-AED cannot emit punctuation: its vocabulary contains no
punctuation tokens at all. Upstream (FireRedASR2S) restores it with a
separate BERT tagger, FireRedPunc, which the mlx-audio port does not
include. This is that step, running the int8 ONNX export -- a ~100MB
download and a few ms per utterance, no new dependency.

The tagger predicts one label per token (none / comma / period / question /
exclamation). We keep the original string and only splice marks in at token
offsets, so casing, spacing and the text itself are returned untouched.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

_REPO = os.environ.get("OMLX_PUNC_REPO", "42ailab/FireRedPunc-ONNX")

# BERT position embeddings cap at 512; leave room for [CLS]/[SEP]
_MAX_TOKENS = 510

# text that already carries any of these is passed through untouched
_EXISTING = "，。？！,.?!"

# no trailing space: the mark lands before the source's own word gap
_HALFWIDTH = {"，": ",", "。": ".", "？": "?", "！": "!"}

_lock = threading.Lock()
_loaded: tuple | None = None
_unavailable = False


def _load():
    """Return (tokenizer, session, labels), or None if the model is absent."""
    global _loaded, _unavailable
    if _loaded is not None or _unavailable:
        return _loaded

    with _lock:
        if _loaded is not None or _unavailable:
            return _loaded
        try:
            import onnxruntime as ort
            from huggingface_hub import snapshot_download
            from tokenizers import Tokenizer

            path = snapshot_download(_REPO)
            tokenizer = Tokenizer.from_file(f"{path}/tokenizer.json")
            session = ort.InferenceSession(
                f"{path}/punc.int8.onnx", providers=["CPUExecutionProvider"]
            )
            with open(f"{path}/out_dict", encoding="utf-8") as f:
                labels = [line.split()[0] for line in f if line.strip()]
            _loaded = (tokenizer, session, labels)
            logger.info("FireRedPunc loaded from %s", path)
        except Exception as exc:
            # fail open: unpunctuated text beats no transcript
            logger.warning("punctuation restoration unavailable: %s", exc)
            _unavailable = True
        return _loaded


def _marks(tokenizer, session, labels, text: str, base: int) -> list[tuple[int, str]]:
    """Predict marks for one window; returns (char offset, mark) pairs."""
    import numpy as np

    enc = tokenizer.encode(text)
    ids = np.array([enc.ids], dtype=np.int64)
    mask = np.array([enc.attention_mask], dtype=np.int64)
    pred = session.run(None, {"input_ids": ids, "attention_mask": mask})[0][0].argmax(-1)

    # logits drop [CLS], so pred[i] labels tokens[i + 1]
    out = []
    for i, label_id in enumerate(pred):
        label = labels[label_id]
        if label == "<space>":
            continue
        start, end = enc.offsets[i + 1]
        if end > start:  # [SEP] carries an empty span
            out.append((base + end, label))
    return out


def punctuate(text: str) -> str:
    """Insert punctuation into unpunctuated text. Returns input on any failure."""
    if not text or any(c in _EXISTING for c in text):
        return text

    loaded = _load()
    if loaded is None:
        return text
    tokenizer, session, labels = loaded

    try:
        marks: list[tuple[int, str]] = []
        enc = tokenizer.encode(text)
        # windowing is for the rare long dictation; one pass covers ~all of them
        if len(enc.ids) <= _MAX_TOKENS:
            marks = _marks(tokenizer, session, labels, text, 0)
        else:
            spans = [(s, e) for s, e in enc.offsets if e > s]
            width = _MAX_TOKENS - 2
            for i in range(0, len(spans), width):
                window = spans[i : i + width]
                start, end = window[0][0], window[-1][1]
                marks += _marks(
                    tokenizer, session, labels, text[start:end], start
                )

        if not marks:
            return text

        ascii_only = text.isascii()
        out, prev = [], 0
        for offset, mark in marks:
            out.append(text[prev:offset])
            out.append(_HALFWIDTH[mark] if ascii_only else mark)
            # a full-width mark already carries its own trailing space
            prev = offset + 1 if not ascii_only and text[offset : offset + 1] == " " else offset
        out.append(text[prev:])
        return "".join(out).strip()
    except Exception as exc:
        logger.warning("punctuation restoration failed: %s", exc)
        return text
