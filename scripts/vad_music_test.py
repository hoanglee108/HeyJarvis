"""Measure the energy gate against Silero VAD on speech mixed with music.

priority.md P0: no change to the audio front end should be believed without numbers,
and this runs entirely offline - no microphone, no speakers.

The speech is synthesised with the project's own TTS model, so the voice matches what
Jarvis is actually tuned for. The music bed is either a real file (``--music``) or a
synthetic stand-in: layered harmonic tones with vibrato plus percussive noise bursts.
Real music with vocals is the harder case, which is exactly why Silero alone is not
expected to solve it - see the ducking section of priority.md.

Usage
-----
    python scripts/vad_music_test.py
    python scripts/vad_music_test.py --music path/to/song.wav --snr 0 5 10
    python scripts/vad_music_test.py --no-stt          # skip transcription
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.audio import FRAME_SAMPLES, read_wav, record_utterance  # noqa: E402
from jarvis.config import load_config  # noqa: E402
from jarvis.console import enable_utf8_output  # noqa: E402
from jarvis.vad import EnergyGate, SileroGate, frame_rms  # noqa: E402

enable_utf8_output()

SAMPLE_RATE = 16000
DEFAULT_SENTENCE = "Jarvis ơi, mở giúp tôi thư mục tải xuống rồi cho tôi biết mấy giờ rồi."


class FrameReader:
    """Feeds a fixed signal to ``record_utterance`` through the MicStream interface."""

    def __init__(self, samples: np.ndarray) -> None:
        usable = samples.size - (samples.size % FRAME_SAMPLES)
        self._frames = samples[:usable].reshape(-1, FRAME_SAMPLES)
        self.index = 0
        self.running = True

    def read(self, timeout: float = 1.0) -> np.ndarray | None:  # noqa: ARG002
        if self.index >= len(self._frames):
            return None
        frame = self._frames[self.index]
        self.index += 1
        return frame

    def recent(self, milliseconds: int) -> np.ndarray:  # noqa: ARG002
        return np.zeros(0, dtype=np.int16)

    def drain(self) -> None:
        return None


# --------------------------------------------------------------------------------------
# signal construction
# --------------------------------------------------------------------------------------
def synth_music(seconds: float, seed: int = 7) -> np.ndarray:
    """A stand-in for instrumental music: chords with vibrato plus a drum pulse."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
    signal = np.zeros_like(t)

    # Chord progression, one chord per bar.
    chords = [(220.0, 277.2, 329.6), (196.0, 246.9, 293.7), (174.6, 220.0, 261.6)]
    bar = 2.0
    for index, chord in enumerate(chords * (int(seconds // (bar * len(chords))) + 1)):
        start, end = index * bar, min(seconds, (index + 1) * bar)
        if start >= seconds:
            break
        mask = (t >= start) & (t < end)
        vibrato = 1.0 + 0.004 * np.sin(2 * np.pi * 5.5 * t)
        for harmonic, weight in ((1, 1.0), (2, 0.35), (3, 0.18)):
            for note in chord:
                signal[mask] += weight * np.sin(
                    2 * np.pi * note * harmonic * t[mask] * vibrato[mask]
                )

    # Percussion: short noise bursts on the beat.
    for beat_start in np.arange(0.0, seconds, 0.5):
        idx = int(beat_start * SAMPLE_RATE)
        length = int(0.06 * SAMPLE_RATE)
        burst = rng.normal(0.0, 1.0, length) * np.exp(-np.linspace(0, 8, length))
        signal[idx : idx + length] += burst[: max(0, min(length, signal.size - idx))] * 3.0

    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak).astype(np.float32)


def load_music(path: Path | None, seconds: float) -> tuple[np.ndarray, str]:
    if path is None:
        return synth_music(seconds), "nhạc tổng hợp (không lời)"
    music, _ = read_wav(path, target_rate=SAMPLE_RATE)
    if music.size < int(seconds * SAMPLE_RATE):
        repeats = int(np.ceil(seconds * SAMPLE_RATE / max(1, music.size)))
        music = np.tile(music, repeats)
    return music[: int(seconds * SAMPLE_RATE)].astype(np.float32), f"nhạc thật ({path.name})"


def synth_speech(sentence: str) -> np.ndarray:
    """Vietnamese speech from the project's configured TTS voice."""
    from jarvis.audio import _resample  # noqa: PLC0415
    from jarvis.tts import TextToSpeech

    config = load_config(PROJECT_ROOT / "config.yaml")
    tts = TextToSpeech(config.tts, speaker=None)  # type: ignore[arg-type]
    result = tts.synthesize(sentence)
    audio = np.asarray(result.samples, dtype=np.float32)
    if result.sample_rate != SAMPLE_RATE:
        audio = np.asarray(_resample(audio, result.sample_rate, SAMPLE_RATE), dtype=np.float32)
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * 0.6).astype(np.float32)


def build_timeline(
    speech: np.ndarray,
    music: np.ndarray,
    snr_db: float,
    lead_silence: float = 1.5,
    tail_silence: float = 3.5,
) -> tuple[np.ndarray, float, float]:
    """Music for the whole clip, speech starting after ``lead_silence`` seconds.

    ``tail_silence`` must comfortably exceed ``audio.silence_timeout_ms``, otherwise the
    clip runs out before the recorder can conclude the turn ended and every gate reports
    ``mic_timeout`` regardless of quality.
    """
    lead = int(lead_silence * SAMPLE_RATE)
    total = lead + speech.size + int(tail_silence * SAMPLE_RATE)
    bed = np.tile(music, int(np.ceil(total / max(1, music.size))))[:total].astype(np.float32)

    speech_rms = float(np.sqrt(np.mean(np.square(speech)))) or 1e-6
    music_rms = float(np.sqrt(np.mean(np.square(bed)))) or 1e-6
    target_music_rms = speech_rms / (10 ** (snr_db / 20))
    bed *= target_music_rms / music_rms

    mixed = bed.copy()
    mixed[lead : lead + speech.size] += speech
    mixed = np.clip(mixed, -1.0, 1.0)
    pcm = (mixed * 32767).astype(np.int16)
    return pcm, lead_silence, lead_silence + speech.size / SAMPLE_RATE


# --------------------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------------------
def frame_scores(pcm: np.ndarray, gate) -> list[bool]:
    usable = pcm.size - (pcm.size % FRAME_SAMPLES)
    gate.reset()
    return [
        bool(gate.is_speech(frame))
        for frame in pcm[:usable].reshape(-1, FRAME_SAMPLES)
    ]


def gate_report(name: str, pcm: np.ndarray, gate, speech_start: float, speech_end: float) -> dict:
    decisions = frame_scores(pcm, gate)
    frame_seconds = FRAME_SAMPLES / SAMPLE_RATE
    inside = [
        index
        for index, _ in enumerate(decisions)
        if speech_start <= index * frame_seconds < speech_end
    ]
    outside = [index for index, _ in enumerate(decisions) if index not in set(inside)]

    hit = sum(decisions[i] for i in inside) / max(1, len(inside))
    false_positive = sum(decisions[i] for i in outside) / max(1, len(outside))
    first_true = next((i for i, flag in enumerate(decisions) if flag), None)
    return {
        "name": name,
        "recall": hit,
        "false_positive": false_positive,
        "onset": None if first_true is None else first_true * frame_seconds,
    }


def segmentation_report(pcm: np.ndarray, config, gate) -> dict:
    reader = FrameReader(pcm)
    gate.reset()
    result = record_utterance(reader, config, gate=gate)  # type: ignore[arg-type]
    return {"reason": result.reason, "duration": result.duration, "samples": result.samples}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--music", type=Path, default=None, help="WAV nhạc thật (mặc định: tổng hợp)")
    parser.add_argument("--snr", type=float, nargs="+", default=[15.0, 5.0, 0.0, -5.0])
    parser.add_argument("--sentence", default=DEFAULT_SENTENCE)
    parser.add_argument("--no-stt", action="store_true", help="Bỏ qua bước nhận dạng")
    parser.add_argument("--save-dir", type=Path, default=None, help="Ghi các bản trộn ra WAV")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "config.yaml")
    audio_config = config.audio.model_copy(deep=True)
    audio_config.max_record_seconds = 30.0
    audio_config.start_timeout_seconds = 30.0

    print("Đang tổng hợp câu nói mẫu bằng model TTS của project…")
    speech = synth_speech(args.sentence)
    music, music_label = load_music(args.music, seconds=8.0)
    print(f"Câu nói: {speech.size / SAMPLE_RATE:.2f}s | Nền: {music_label}\n")

    stt = None
    if not args.no_stt:
        from jarvis.stt import SpeechToText

        stt = SpeechToText(config.stt)
        stt.load()

    energy_threshold = max(
        audio_config.calibration_floor_minimum,
        (audio_config.silence_rms_threshold or 0.0),
    ) or 0.006

    for snr in args.snr:
        pcm, speech_start, speech_end = build_timeline(speech, music, snr)
        music_only = pcm[: int(speech_start * SAMPLE_RATE)]
        print("=" * 78)
        print(
            f"SNR {snr:+.0f} dB  |  RMS nền {frame_rms(music_only):.4f}  "
            f"|  ngưỡng năng lượng {energy_threshold:.4f}"
        )

        if args.save_dir:
            from jarvis.audio import write_wav

            args.save_dir.mkdir(parents=True, exist_ok=True)
            write_wav(args.save_dir / f"mix_snr{int(snr):+d}.wav", pcm, SAMPLE_RATE)

        gates = {
            "energy": EnergyGate(energy_threshold),
            "silero": SileroGate(
                threshold=audio_config.vad_threshold, min_rms=audio_config.vad_min_rms
            ),
        }

        print(f"\n  {'gate':10} {'recall':>8} {'kích sai':>10} {'onset':>8}   phân đoạn")
        for name, gate in gates.items():
            scores = gate_report(name, pcm, gate, speech_start, speech_end)
            seg = segmentation_report(pcm, audio_config, gate)
            onset = "-" if scores["onset"] is None else f"{scores['onset']:.2f}s"
            print(
                f"  {name:10} {scores['recall'] * 100:7.1f}% {scores['false_positive'] * 100:9.1f}% "
                f"{onset:>8}   {seg['reason']:<12} {seg['duration']:.2f}s"
            )
            if stt is not None and seg["samples"].size:
                transcript = stt.transcribe(seg["samples"])
                print(f"  {'':10} STT: {transcript[:90]!r}")
        print()

    print("=" * 78)
    print(
        "Đọc kết quả: 'recall' = tỷ lệ frame có giọng được nhận đúng, 'kích sai' = tỷ lệ\n"
        "frame chỉ có nhạc bị coi là giọng. Phân đoạn 'silence' là tốt; 'max_duration'\n"
        "nghĩa là cổng không bao giờ thấy im lặng - đúng lỗi mà P1-2 phải sửa."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
