"""Lower other applications' volume while Jarvis is listening (priority.md P1-3).

Why per-application and not the master volume
---------------------------------------------
Jarvis's own text-to-speech goes out of the same speakers as the music. Dropping the
master volume would mute Jarvis together with the music, which is the opposite of what
we want. Windows exposes a *per-session* volume through the audio session API
(``IAudioSessionManager2`` / ``ISimpleAudioVolume``), so the browser or Spotify can be
attenuated while our own process is left alone - it is skipped by PID.

This is not a shortcut around echo cancellation; it is what smart speakers do as well.
Echo cancellation only has to get the *wake word* through the music, and the command
itself is captured with the content turned down.

Deliberate limits
-----------------
* **Never ducks while waiting for the wake word.** ``Pipeline._capture`` serves both the
  standby listen and the command listen; ducking the former would hold the music down
  forever. The caller decides via the ``duck`` argument.
* **Nothing here may break the pipeline.** Every COM call is wrapped; if pycaw is
  missing or a session disappears mid-call, ducking silently turns itself off.
* **Restoring is asynchronous.** A 250 ms fade on the pipeline thread would be 250 ms
  added to every turn's latency, so the ramp runs on a daemon thread and is abandoned if
  a new duck starts while it is still fading up.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

from .config import DuckingConfig
from .logging_setup import get_logger

log = get_logger("jarvis.ducking")

#: Volume steps per second while fading back up.
_RAMP_STEP_MS = 50


class VolumeDucker:
    """Reference-counted, per-application volume attenuation.

    ``acquire``/``release`` nest, so overlapping listen and speak phases do not fight
    over the volume: the music only comes back once the last holder releases.
    """

    def __init__(self, config: DuckingConfig) -> None:
        self._config = config
        self._lock = threading.RLock()
        self._depth = 0
        self._generation = 0
        #: ``(ISimpleAudioVolume, original level)`` captured on the first duck of a cycle
        #: and kept until a fade-up finishes, so an abandoned fade cannot be mistaken for
        #: the user's real volume.
        self._ducked: list[tuple[Any, float]] = []
        self._com_threads: set[int] = set()
        #: ``None`` = not probed yet, ``False`` = unavailable, stop trying.
        self._available: bool | None = None if config.enabled else False

    # -- availability ------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._config.enabled and self._available is not False

    @property
    def active(self) -> bool:
        with self._lock:
            return bool(self._ducked)

    def _init_com(self) -> None:
        """COM must be initialised per thread; the pipeline calls us from several."""
        thread_id = threading.get_ident()
        if thread_id in self._com_threads:
            return
        try:
            import comtypes

            comtypes.CoInitialize()
        except Exception:  # pragma: no cover - already initialised is fine
            log.debug("CoInitialize không thành công", exc_info=True)
        self._com_threads.add(thread_id)

    def _sessions(self) -> list[tuple[Any, str, int]]:
        """``(volume interface, process name, pid)`` for every live audio session."""
        self._init_com()
        from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume

        found: list[tuple[Any, str, int]] = []
        for session in AudioUtilities.GetAllSessions():
            try:
                process = session.Process
                if process is None:
                    # PID 0 is the system-sounds session; leave it alone.
                    continue
                volume = session._ctl.QueryInterface(ISimpleAudioVolume)
                found.append((volume, (process.name() or "").lower(), int(session.ProcessId)))
            except Exception:  # a session can vanish between enumeration and use
                log.debug("Bỏ qua một audio session không đọc được", exc_info=True)
        return found

    def _wanted(self, name: str, pid: int) -> bool:
        if pid == os.getpid():
            return False  # never duck our own text-to-speech
        if name in self._config.exclude_processes:
            return False
        if self._config.processes:
            return name in self._config.processes
        return True

    # -- ducking -----------------------------------------------------------------
    def acquire(self) -> None:
        """Duck now (idempotent while held); pair with :meth:`release`."""
        if not self.enabled:
            return
        with self._lock:
            self._depth += 1
            if self._depth > 1:
                return
            self._generation += 1
            if self._ducked:
                # A fade-up was still running: re-apply to the sessions we already own
                # rather than re-reading their (mid-fade) level as the original.
                self._set_all(self._config.level)
                return
            try:
                candidates = [
                    (volume, name)
                    for volume, name, pid in self._sessions()
                    if self._wanted(name, pid)
                ]
            except ImportError:
                self._available = False
                log.warning(
                    "Chưa cài pycaw nên không hạ được âm lượng khi nghe: "
                    "pip install -r requirements.txt"
                )
                return
            except Exception as exc:
                self._available = False
                log.warning("Không dùng được ducking âm lượng (%s); bỏ qua.", exc)
                return

            self._available = True
            ducked: list[tuple[Any, float]] = []
            names: list[str] = []
            for volume, name in candidates:
                try:
                    current = float(volume.GetMasterVolume())
                    if current <= self._config.level + 1e-3:
                        continue  # already quiet; nothing to restore later
                    volume.SetMasterVolume(self._config.level, None)
                except Exception:
                    log.debug("Không hạ được âm lượng của %s", name, exc_info=True)
                    continue
                ducked.append((volume, current))
                names.append(name)
            self._ducked = ducked
            if ducked:
                log.info(
                    "Đã hạ âm lượng xuống %.0f%%: %s",
                    self._config.level * 100,
                    ", ".join(sorted(set(names))),
                )

    def release(self) -> None:
        """Drop one hold; the last one starts the fade back up."""
        with self._lock:
            if self._depth == 0:
                return
            self._depth -= 1
            if self._depth > 0 or not self._ducked:
                return
            self._generation += 1
            generation = self._generation
            targets = list(self._ducked)

        if self._config.restore_ms <= 0:
            self._finish_restore(targets, generation)
            return
        threading.Thread(
            target=self._fade_up,
            args=(targets, generation),
            name="jarvis-unduck",
            daemon=True,
        ).start()

    def restore_all(self) -> None:
        """Put every volume back immediately; used on shutdown.

        Without this, crashing while ducked would leave the user's music quiet. Session
        volumes are not persistent, so restarting the app would also fix it, but that is
        not a good thing to make somebody discover.
        """
        with self._lock:
            self._depth = 0
            # Bumping the generation makes any in-flight fade-up abandon itself.
            self._generation += 1
            targets, self._ducked = self._ducked, []
        for volume, original in targets:
            try:
                volume.SetMasterVolume(original, None)
            except Exception:  # pragma: no cover - shutdown must not raise
                log.debug("Không phục hồi được âm lượng", exc_info=True)
        if targets:
            log.info("Đã phục hồi âm lượng của %d session", len(targets))

    # -- internals ---------------------------------------------------------------
    def _set_all(self, level: float) -> None:
        for volume, _original in self._ducked:
            try:
                volume.SetMasterVolume(level, None)
            except Exception:
                log.debug("Không đặt được âm lượng session", exc_info=True)

    def _fade_up(self, targets: list[tuple[Any, float]], generation: int) -> None:
        self._init_com()
        steps = max(1, int(self._config.restore_ms / _RAMP_STEP_MS))
        pause = self._config.restore_ms / 1000.0 / steps
        for step in range(1, steps + 1):
            fraction = step / steps
            with self._lock:
                if self._generation != generation:
                    return  # a new duck took over; leave the level where it wants it
                for volume, original in targets:
                    try:
                        volume.SetMasterVolume(
                            self._config.level + (original - self._config.level) * fraction,
                            None,
                        )
                    except Exception:
                        log.debug("Không tăng lại được âm lượng", exc_info=True)
            if step < steps:
                time.sleep(pause)
        self._finish_restore(targets, generation)

    def _finish_restore(self, targets: list[tuple[Any, float]], generation: int) -> None:
        with self._lock:
            if self._generation != generation:
                return
            for volume, original in targets:
                try:
                    volume.SetMasterVolume(original, None)
                except Exception:
                    log.debug("Không phục hồi được âm lượng", exc_info=True)
            self._ducked = []
        log.debug("Đã phục hồi âm lượng (%d session)", len(targets))

    # -- convenience -------------------------------------------------------------
    @contextmanager
    def ducked(self, *, active: bool = True) -> Iterator[None]:
        """Duck for the duration of the block when ``active``."""
        if not active or not self.enabled:
            yield
            return
        self.acquire()
        try:
            yield
        finally:
            self.release()
