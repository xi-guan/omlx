# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SenseVoice ITN (inverse text normalization) gating.

ITN adds punctuation and normalizes numbers/dates. STTEngine enables it by
default for SenseVoice models only, since other backends (whisper, voxtral,
qwen3-asr) do not accept ``use_itn``. All tests run with a FakeModel — no
mlx-audio dependency.
"""

import io
import wave
from types import SimpleNamespace

import pytest


def _make_wav_bytes(duration_secs: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Generate minimal valid WAV bytes (silence)."""
    n_samples = int(sample_rate * duration_secs)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


TINY_WAV = _make_wav_bytes()


def _capture_model() -> tuple[object, dict]:
    """FakeModel that records the kwargs passed to generate()."""
    captured: dict = {}

    class FakeModel:
        def generate(self, audio_path, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                text="hello", language=None, segments=[], total_time=0.1
            )

    return FakeModel(), captured


@pytest.mark.asyncio
async def test_sensevoice_enables_itn_by_default(tmp_path, monkeypatch):
    """SenseVoice models get use_itn=True injected when not set by the env."""
    from omlx.engine.stt import STTEngine

    monkeypatch.delenv("OMLX_SENSEVOICE_ITN", raising=False)
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(TINY_WAV)

    model, captured = _capture_model()
    engine = STTEngine("SenseVoiceSmall")
    engine._model = model

    await engine.transcribe(str(audio_path))

    assert captured.get("use_itn") is True


@pytest.mark.asyncio
async def test_non_sensevoice_never_gets_itn(tmp_path):
    """Whisper/Qwen models must not receive use_itn — they reject it."""
    from omlx.engine.stt import STTEngine

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(TINY_WAV)

    model, captured = _capture_model()
    engine = STTEngine("whisper-tiny")
    engine._model = model

    await engine.transcribe(str(audio_path))

    assert "use_itn" not in captured


@pytest.mark.asyncio
async def test_env_var_disables_itn(tmp_path, monkeypatch):
    """OMLX_SENSEVOICE_ITN=0 turns the default off."""
    from omlx.engine.stt import STTEngine

    monkeypatch.setenv("OMLX_SENSEVOICE_ITN", "0")
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(TINY_WAV)

    model, captured = _capture_model()
    engine = STTEngine("SenseVoiceSmall")
    engine._model = model

    await engine.transcribe(str(audio_path))

    assert captured.get("use_itn") is False


@pytest.mark.asyncio
async def test_explicit_use_itn_wins(tmp_path):
    """A caller-provided use_itn overrides the SenseVoice default."""
    from omlx.engine.stt import STTEngine

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(TINY_WAV)

    model, captured = _capture_model()
    engine = STTEngine("SenseVoiceSmall")
    engine._model = model

    await engine.transcribe(str(audio_path), use_itn=False)

    assert captured.get("use_itn") is False
