"""Microphone capture and speaker playback.

Design notes
------------
* One long-lived :class:`MicStream` owns the only ``InputStream`` in the process.
  The wake-word detector and the utterance recorder both pull 80 ms int16 frames
  from the same queue, so switching from "listening for wake word" to "recording
  the command" never reopens the device (no dropouts, no "device busy" races).
* A ring buffer keeps the last ~2 s of audio so the recorder can prepend pre-roll
  and not clip the first syllable.
* If the device refuses 16 kHz, we open at its native rate and downsample with
  ``scipy.signal.resample_poly`` (anti-aliased) instead of failing.
"""

from __future__ import annotations

import math
import queue
import threading
from collections import deque
from pathlib import Path

import numpy as np

from .config import AudioConfig
from .logging_setup import get_logger

log = get_logger("jarvis.audio")

INT16_MAX = 32768.0
#: openWakeWord operates on 80 ms frames.
FRAME_SAMPLES = 1280


class AudioError(RuntimeError):
    """Microphone or speaker could not be used."""


# --------------------------------------------------------------------------------------
# Device helpers
# --------------------------------------------------------------------------------------
def _sd():
    try:
        import sounddevice as sd
    except OSError as exc:  # pragma: no cover - missing PortAudio DLL
        raise AudioError(f"Không nạp được PortAudio (sounddevice): {exc}") from exc
    return sd


def describe_devices() -> str:
    """Human-readable device table for ``python -m jarvis devices``."""
    sd = _sd()
    lines: list[str] = []
    try:
        default_in, default_out = sd.default.device
    except Exception:  # pragma: no cover
        default_in = default_out = None
    for index, dev in enumerate(sd.query_devices()):
        flags = []
        if dev["max_input_channels"] > 0:
            flags.append("in")
        if dev["max_output_channels"] > 0:
            flags.append("out")
        marks = []
        if index == default_in:
            marks.append("default-in")
        if index == default_out:
            marks.append("default-out")
        lines.append(
            f"[{index:>2}] {dev['name']}  ({'/'.join(flags) or '-'}, "
            f"{int(dev['default_samplerate'])} Hz)"
            + (f"  <- {', '.join(marks)}" if marks else "")
        )
    return "\n".join(lines)


def resolve_device(spec: int | str | None, *, want_input: bool) -> int | None:
    """Turn ``None`` / index / name-substring into a PortAudio device index."""
    if spec is None:
        return None
    sd = _sd()
    devices = sd.query_devices()
    if isinstance(spec, int):
        if not 0 <= spec < len(devices):
            raise AudioError(f"Không có thiết bị audio index {spec}. Chạy `python -m jarvis devices`.")
        return spec

    needle = str(spec).strip().lower()
    channel_key = "max_input_channels" if want_input else "max_output_channels"
    for index, dev in enumerate(devices):
        if dev[channel_key] > 0 and needle in dev["name"].lower():
            return index
    kind = "input" if want_input else "output"
    raise AudioError(
        f"Không tìm thấy thiết bị {kind} nào khớp {spec!r}. Chạy `python -m jarvis devices`."
    )


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Anti-aliased resampling; falls back to linear interpolation if SciPy is absent."""
    if src_rate == dst_rate or samples.size == 0:
        return samples
    try:
        from scipy.signal import resample_poly

        gcd = math.gcd(src_rate, dst_rate)
        return resample_poly(samples.astype(np.float32), dst_rate // gcd, src_rate // gcd)
    except Exception:  # pragma: no cover - SciPy always ships with openwakeword
        target_len = int(round(samples.size * dst_rate / src_rate))
        src_x = np.linspace(0.0, 1.0, samples.size, endpoint=False)
        dst_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
        return np.interp(dst_x, src_x, samples.astype(np.float32)).astype(np.float32)


def rms(frame: np.ndarray) -> float:
    """Root-mean-square of an int16 or float frame, normalised to 0..1."""
    if frame.size == 0:
        return 0.0
    data = frame.astype(np.float32)
    if np.issubdtype(frame.dtype, np.integer):
        data /= INT16_MAX
    return float(np.sqrt(np.mean(np.square(data))))


# --------------------------------------------------------------------------------------
# Microphone stream
# --------------------------------------------------------------------------------------
class MicStream:
    """Continuous microphone capture producing fixed-size int16 frames at 16 kHz."""

    def __init__(self, config: AudioConfig, *, frame_samples: int = FRAME_SAMPLES) -> None:
        self._config = config
        self._target_rate = config.sample_rate
        self._frame_samples = frame_samples
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._ring: deque[np.ndarray] = deque(maxlen=max(1, int(2.0 * config.sample_rate / frame_samples)))
        self._ring_lock = threading.Lock()
        self._stream = None
        self._native_rate = config.sample_rate
        self._native_block = frame_samples
        self._overflow_count = 0

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> None:
        if self._stream is not None:
            return
        sd = _sd()
        device = resolve_device(self._config.input_device, want_input=True)

        candidates: list[int] = [self._target_rate]
        try:
            info = sd.query_devices(device if device is not None else sd.default.device[0], "input")
            native = int(info["default_samplerate"])
            if native != self._target_rate:
                candidates.append(native)
        except Exception:  # pragma: no cover - best effort
            pass
        candidates.extend(rate for rate in (48000, 44100) if rate not in candidates)

        last_error: Exception | None = None
        for rate in candidates:
            block = int(round(self._frame_samples * rate / self._target_rate))
            try:
                stream = sd.InputStream(
                    samplerate=rate,
                    blocksize=block,
                    device=device,
                    channels=1,
                    dtype="int16",
                    callback=self._callback,
                )
                stream.start()
            except Exception as exc:  # PortAudioError and friends
                last_error = exc
                log.debug("Mở mic ở %d Hz thất bại: %s", rate, exc)
                continue
            self._stream = stream
            self._native_rate = rate
            self._native_block = block
            log.info(
                "Mic đang mở: device=%s, %d Hz%s",
                device if device is not None else "default",
                rate,
                "" if rate == self._target_rate else f" (resample -> {self._target_rate} Hz)",
            )
            return

        raise AudioError(
            "Không mở được micro. Kiểm tra thiết bị và quyền truy cập microphone của Windows. "
            f"Lỗi cuối: {last_error}"
        )

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # pragma: no cover
                log.debug("Lỗi khi đóng mic stream", exc_info=True)

    def __enter__(self) -> MicStream:
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()

    @property
    def running(self) -> bool:
        return self._stream is not None

    # -- capture -----------------------------------------------------------------
    def _callback(self, indata, _frames, _time, status) -> None:  # pragma: no cover - realtime
        if status and status.input_overflow:
            self._overflow_count += 1
            if self._overflow_count % 25 == 1:
                log.debug("Mic input overflow (%d lần)", self._overflow_count)
        # PortAudio reuses ``indata`` for the next callback, so the frame must be
        # copied. Without this, every queued/ring frame aliases one buffer and the
        # recorded utterance is overwritten by later audio before it reaches STT.
        mono = np.array(indata, dtype=np.int16, copy=True).reshape(-1)
        if self._native_rate != self._target_rate:
            resampled = _resample(mono.astype(np.float32), self._native_rate, self._target_rate)
            mono = np.clip(resampled, -INT16_MAX, INT16_MAX - 1).astype(np.int16)
        with self._ring_lock:
            self._ring.append(mono)
        try:
            self._queue.put_nowait(mono)
        except queue.Full:
            # Drop the oldest frame: staying real-time matters more than completeness.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(mono)
            except queue.Empty:  # pragma: no cover
                pass

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        """Next captured frame (int16, ``frame_samples`` long) or ``None`` on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        """Discard queued frames (e.g. audio captured while Jarvis was speaking)."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def recent(self, milliseconds: int) -> np.ndarray:
        """Most recent audio from the ring buffer, for pre-roll."""
        if milliseconds <= 0:
            return np.zeros(0, dtype=np.int16)
        wanted = int(self._target_rate * milliseconds / 1000)
        with self._ring_lock:
            frames = list(self._ring)
        if not frames:
            return np.zeros(0, dtype=np.int16)
        buffer = np.concatenate(frames)
        return buffer[-wanted:] if buffer.size > wanted else buffer


# --------------------------------------------------------------------------------------
# Utterance recording
# --------------------------------------------------------------------------------------
class RecordingResult:
    __slots__ = ("samples", "sample_rate", "reason", "peak_rms")

    def __init__(self, samples: np.ndarray, sample_rate: int, reason: str, peak_rms: float) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.reason = reason
        self.peak_rms = peak_rms

    @property
    def duration(self) -> float:
        return self.samples.size / self.sample_rate if self.sample_rate else 0.0

    @property
    def has_speech(self) -> bool:
        return self.reason in ("silence", "max_duration")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"RecordingResult(duration={self.duration:.2f}s, reason={self.reason!r}, "
            f"peak_rms={self.peak_rms:.4f})"
        )


def calibrate_noise_floor(mic: MicStream, config: AudioConfig) -> float:
    """Listen to ambient noise and derive a speech/silence RMS threshold."""
    if config.silence_rms_threshold is not None:
        return float(config.silence_rms_threshold)

    frames_needed = max(1, int(config.calibration_seconds * config.sample_rate / FRAME_SAMPLES))
    levels: list[float] = []
    for _ in range(frames_needed):
        frame = mic.read(timeout=1.0)
        if frame is None:
            break
        levels.append(rms(frame))

    noise_floor = float(np.median(levels)) if levels else 0.0
    threshold = max(noise_floor * config.calibration_multiplier, config.calibration_floor_minimum)
    log.info("Hiệu chuẩn tiếng ồn: nền=%.5f -> ngưỡng nói=%.5f", noise_floor, threshold)
    return threshold


def record_utterance(
    mic: MicStream,
    config: AudioConfig,
    *,
    speech_threshold: float,
    pre_roll: np.ndarray | None = None,
) -> RecordingResult:
    """Record until the speaker goes quiet (VAD-style energy gate).

    Stop reasons: ``silence`` (normal), ``max_duration``, ``no_speech`` (nobody
    spoke within ``start_timeout_seconds``), ``mic_timeout``.
    """
    frame_seconds = FRAME_SAMPLES / config.sample_rate
    silence_frames_needed = max(1, int((config.silence_timeout_ms / 1000) / frame_seconds))
    max_frames = int(config.max_record_seconds / frame_seconds)
    min_speech_frames = max(1, int(config.min_record_seconds / frame_seconds))
    start_deadline_frames = int(config.start_timeout_seconds / frame_seconds)

    collected: list[np.ndarray] = []
    if pre_roll is not None and pre_roll.size:
        collected.append(pre_roll.astype(np.int16))

    # Frames captured while waiting for the user to start talking; the last few are
    # kept so the leading consonant is not lost.
    lead_in: deque[np.ndarray] = deque(maxlen=4)

    speech_frames = 0
    silence_run = 0
    frames_seen = 0
    peak = 0.0
    started = False
    reason = "max_duration"

    while frames_seen < max_frames:
        frame = mic.read(timeout=2.0)
        if frame is None:
            reason = "mic_timeout"
            break
        frames_seen += 1
        level = rms(frame)
        peak = max(peak, level)
        is_speech = level >= speech_threshold

        if not started:
            if is_speech:
                started = True
                collected.extend(lead_in)
                collected.append(frame)
                speech_frames += 1
            elif frames_seen >= start_deadline_frames:
                reason = "no_speech"
                break
            else:
                lead_in.append(frame)
            continue

        collected.append(frame)
        if is_speech:
            speech_frames += 1
            silence_run = 0
        else:
            silence_run += 1
            if silence_run >= silence_frames_needed and speech_frames >= min_speech_frames:
                reason = "silence"
                break

    samples = (
        np.concatenate(collected) if collected else np.zeros(0, dtype=np.int16)
    ).astype(np.int16)
    result = RecordingResult(samples, config.sample_rate, reason, peak)
    log.info(
        "Ghi âm xong: %.2fs, lý do=%s, peak_rms=%.4f (ngưỡng %.4f)",
        result.duration,
        reason,
        peak,
        speech_threshold,
    )
    return result


def record_fixed(config: AudioConfig, seconds: float) -> np.ndarray:
    """Blocking fixed-length recording (used by ``jarvis record``)."""
    sd = _sd()
    device = resolve_device(config.input_device, want_input=True)
    frames = int(seconds * config.sample_rate)
    try:
        data = sd.rec(
            frames,
            samplerate=config.sample_rate,
            channels=1,
            dtype="int16",
            device=device,
        )
        sd.wait()
    except Exception as exc:
        raise AudioError(f"Không ghi âm được: {exc}") from exc
    return np.asarray(data, dtype=np.int16).reshape(-1)


# --------------------------------------------------------------------------------------
# Playback
# --------------------------------------------------------------------------------------
class Speaker:
    """Blocking playback of float32/int16 mono audio."""

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._device: int | None = None
        self._device_resolved = False
        self._lock = threading.Lock()

    def _resolved_device(self) -> int | None:
        if not self._device_resolved:
            self._device = resolve_device(self._config.output_device, want_input=False)
            self._device_resolved = True
        return self._device

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        if samples.size == 0:
            return
        sd = _sd()
        data = np.asarray(samples)
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / INT16_MAX
        data = np.clip(data.astype(np.float32), -1.0, 1.0)
        with self._lock:
            try:
                sd.play(data, samplerate=sample_rate, device=self._resolved_device())
                sd.wait()
            except Exception as exc:
                raise AudioError(f"Không phát được audio: {exc}") from exc

    def stop(self) -> None:
        try:
            _sd().stop()
        except Exception:  # pragma: no cover
            pass

    def beep(self, frequency: float = 880.0, duration: float = 0.12, volume: float = 0.25) -> None:
        """Short sine cue so the user knows Jarvis started listening."""
        sample_rate = 22050
        t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
        envelope = np.minimum(1.0, np.minimum(t, duration - t) * 40.0)
        tone = np.sin(2 * np.pi * frequency * t) * volume * envelope
        try:
            self.play(tone.astype(np.float32), sample_rate)
        except AudioError:  # a missing beep must never break the pipeline
            log.debug("Không phát được beep", exc_info=True)


# --------------------------------------------------------------------------------------
# WAV helpers
# --------------------------------------------------------------------------------------
def read_wav(path: str | Path, target_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Read a WAV file as mono float32 in ``[-1, 1]``, resampled to ``target_rate``."""
    import soundfile as sf

    wav_path = Path(path)
    if not wav_path.is_file():
        raise AudioError(f"Không tìm thấy file audio: {wav_path}")
    data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    if sample_rate != target_rate:
        mono = _resample(mono, sample_rate, target_rate).astype(np.float32)
        sample_rate = target_rate
    return mono, sample_rate


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    import soundfile as sf

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(samples)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / INT16_MAX
    sf.write(str(out), np.clip(data, -1.0, 1.0), sample_rate, subtype="PCM_16")
    return out


def to_float32(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples)
    if np.issubdtype(data.dtype, np.integer):
        return (data.astype(np.float32) / INT16_MAX).reshape(-1)
    return data.astype(np.float32).reshape(-1)
