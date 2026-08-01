# SPDX-License-Identifier: Apache-2.0
"""Unit tests for punctuation restoration and its gating.

FireRedASR2-AED has no punctuation in its vocabulary, so STTEngine runs the
FireRedPunc tagger over its output. Gating tests use a FakeModel and a stubbed
tagger, so they need neither mlx-audio nor the 100MB ONNX download; the
splicing tests skip when the model is absent.
"""

import io
import wave
from types import SimpleNamespace

import pytest


def _make_wav_bytes(duration_secs: float = 0.1, sample_rate: int = 16000) -> bytes:
    n_samples = int(sample_rate * duration_secs)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


TINY_WAV = _make_wav_bytes()


def _engine_with(model_name: str, text: str, tmp_path, monkeypatch):
    """STTEngine wired to a FakeModel, with punctuate() stubbed to a marker."""
    from omlx.engine import punc
    from omlx.engine.stt import STTEngine

    monkeypatch.setattr(punc, "punctuate", lambda t: f"{t}<punctuated>")

    class FakeModel:
        def generate(self, audio_path, **kwargs):
            return SimpleNamespace(
                text=text, language=None, segments=[], total_time=0.1
            )

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(TINY_WAV)

    engine = STTEngine(model_name)
    engine._model = FakeModel()
    return engine, str(audio_path)


@pytest.mark.asyncio
async def test_firered_output_is_punctuated(tmp_path, monkeypatch):
    monkeypatch.delenv("OMLX_PUNC", raising=False)
    engine, path = _engine_with("FireRedASR2-AED-mlx", "你好世界", tmp_path, monkeypatch)

    result = await engine.transcribe(path)

    assert result["text"] == "你好世界<punctuated>"


@pytest.mark.asyncio
async def test_other_models_are_untouched(tmp_path, monkeypatch):
    """Qwen3/whisper already punctuate; running the tagger would double up."""
    engine, path = _engine_with("Qwen3-ASR-1.7B-bf16", "你好世界", tmp_path, monkeypatch)

    result = await engine.transcribe(path)

    assert result["text"] == "你好世界"


@pytest.mark.asyncio
async def test_env_var_disables_punc(tmp_path, monkeypatch):
    monkeypatch.setenv("OMLX_PUNC", "0")
    engine, path = _engine_with("FireRedASR2-AED-mlx", "你好世界", tmp_path, monkeypatch)

    result = await engine.transcribe(path)

    assert result["text"] == "你好世界"


@pytest.mark.asyncio
async def test_firered_also_gets_itn(tmp_path, monkeypatch):
    """FireRed emits chinese numerals too, so ITN applies to it as well."""
    monkeypatch.setenv("OMLX_PUNC", "0")
    engine, path = _engine_with("FireRedASR2-AED-mlx", "二零二六年", tmp_path, monkeypatch)

    result = await engine.transcribe(path)

    assert result["text"] == "2026年"


# --- the tagger itself (needs the downloaded model) ------------------------


def _tagger_available() -> bool:
    from omlx.engine.punc import _load

    return _load() is not None


needs_model = pytest.mark.skipif(
    not _tagger_available(), reason="FireRedPunc model not downloaded"
)


@needs_model
@pytest.mark.parametrize(
    "text,expected",
    [
        ("好像确实是标点符号有问题", "好像确实是标点符号有问题。"),
        ("请问这标点符号这问题咋解决呢", "请问这标点符号这问题咋解决呢？"),
        ("this is a test how does it work", "this is a test, how does it work."),
    ],
)
def test_marks_are_spliced_in(text, expected):
    from omlx.engine.punc import punctuate

    assert punctuate(text) == expected


@needs_model
def test_already_punctuated_text_is_left_alone():
    from omlx.engine.punc import punctuate

    text = "已经有标点了。不要再加。"
    assert punctuate(text) == text


@needs_model
def test_only_punctuation_is_added():
    """Everything but the inserted marks must survive verbatim."""
    from omlx.engine.punc import punctuate

    text = "那你觉得有没有必要搞一个 centralized 的 portal 然后呢"
    result = punctuate(text)
    keep = "，。？！ "  # a full-width mark absorbs the space it lands on
    assert "".join(c for c in result if c not in keep) == text.replace(" ", "")


@needs_model
def test_fullwidth_mark_absorbs_the_following_space():
    """"portal， 然后" is wrong typography; the mark carries its own gap."""
    from omlx.engine.punc import punctuate

    assert "， " not in punctuate("我用的是 portal 然后呢就这样")


@needs_model
def test_long_input_is_windowed():
    """Past 512 tokens the tagger must window rather than fail open."""
    from omlx.engine.punc import punctuate

    text = "我今天去了公司然后开了一个会" * 90
    result = punctuate(text)
    assert any(c in result for c in "，。？！")
    assert "".join(c for c in result if c not in "，。？！") == text
