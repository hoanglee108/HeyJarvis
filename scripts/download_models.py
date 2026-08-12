"""Download every model Jarvis needs into ``models/`` (and openWakeWord's own cache).

Usage (from the project root, with the venv active):

    python scripts/download_models.py                # int8 STT + vais1000 voice
    python scripts/download_models.py --full-precision
    python scripts/download_models.py --voice vivos  # lighter voice
    python scripts/download_models.py --only tts

Everything is resumable-ish: existing files with the expected size are skipped.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jarvis.console import enable_utf8_output  # noqa: E402

enable_utf8_output()

MODELS_DIR = PROJECT_ROOT / "models"
STT_DIR = MODELS_DIR / "stt" / "zipformer-30m-rnnt-6000h"
TTS_DIR = MODELS_DIR / "tts"

HF_REPO = "hynt/Zipformer-30M-RNNT-6000h"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"

#: The repo ships the token table as ``config.json``; sherpa-onnx expects ``tokens.txt``.
STT_FILES_INT8 = {
    "encoder-epoch-20-avg-10.int8.onnx": "encoder-epoch-20-avg-10.int8.onnx",
    "decoder-epoch-20-avg-10.onnx": "decoder-epoch-20-avg-10.onnx",
    "joiner-epoch-20-avg-10.int8.onnx": "joiner-epoch-20-avg-10.int8.onnx",
    "config.json": "tokens.txt",
}
STT_FILES_FULL = {
    "encoder-epoch-20-avg-10.onnx": "encoder-epoch-20-avg-10.onnx",
    "joiner-epoch-20-avg-10.onnx": "joiner-epoch-20-avg-10.onnx",
}

TTS_RELEASE_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
VOICES = {
    "vais1000": "vits-piper-vi_VN-vais1000-medium",
    "vivos": "vits-piper-vi_VN-vivos-x_low",
    "25hours": "vits-piper-vi_VN-25hours_single-low",
}
DEFAULT_VOICE = "vais1000"

USER_AGENT = "jarvis-model-downloader/0.1"


# --------------------------------------------------------------------------------------
def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def show(path: Path) -> str:
    """Path relative to the project root when possible, else absolute."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def download(url: str, destination: Path, *, expected_size: int | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = destination.stat().st_size
        if expected_size is None or actual == expected_size:
            print(f"  [skip] {destination.name} ({human(actual)}) đã có")
            return destination
        print(f"  [redo] {destination.name} sai kích thước ({actual} != {expected_size})")

    print(f"  [get ] {url}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    temp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temp_path.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            next_mark = 10
            while chunk := response.read(1 << 18):
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 / total
                    if percent >= next_mark:
                        print(f"         {human(downloaded)} / {human(total)} ({percent:5.1f}%)")
                        next_mark += 10
    except (urllib.error.URLError, OSError) as exc:
        temp_path.unlink(missing_ok=True)
        raise SystemExit(f"Tải thất bại: {url}\n  {exc}") from exc

    temp_path.replace(destination)
    print(f"  [ok  ] {show(destination)} ({human(destination.stat().st_size)})")
    return destination


# --------------------------------------------------------------------------------------
def fetch_stt(full_precision: bool) -> None:
    print(f"\n== STT: {HF_REPO} -> {show(STT_DIR)}")
    wanted = dict(STT_FILES_INT8)
    if full_precision:
        wanted.update(STT_FILES_FULL)
    for remote_name, local_name in wanted.items():
        url = f"{HF_BASE}/{remote_name}"
        download(url, STT_DIR / local_name, expected_size=remote_size(url))

    tokens = STT_DIR / "tokens.txt"
    if tokens.is_file():
        count = sum(1 for line in tokens.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"  tokens.txt: {count} tokens")


def fetch_tts(voice: str) -> None:
    if voice not in VOICES:
        raise SystemExit(f"Giọng không hợp lệ: {voice}. Chọn một trong {sorted(VOICES)}")
    name = VOICES[voice]
    target = TTS_DIR / name
    print(f"\n== TTS: {name} -> {show(target)}")

    model_file = target / f"{name.replace('vits-piper-', '')}.onnx"
    if model_file.is_file() and (target / "tokens.txt").is_file():
        print(f"  [skip] {name} đã có")
        return

    url = f"{TTS_RELEASE_BASE}/{name}.tar.bz2"
    with tempfile.TemporaryDirectory(prefix="jarvis-tts-") as tmp:
        archive = Path(tmp) / f"{name}.tar.bz2"
        download(url, archive, expected_size=remote_size(url))
        print("  [extract]")
        with tarfile.open(archive, "r:bz2") as tar:
            _safe_extract(tar, Path(tmp))
        extracted = Path(tmp) / name
        if not extracted.is_dir():  # some archives nest differently
            candidates = [p for p in Path(tmp).iterdir() if p.is_dir()]
            if not candidates:
                raise SystemExit(f"Không tìm thấy nội dung sau khi giải nén {archive.name}")
            extracted = candidates[0]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(extracted), str(target))

    onnx_files = sorted(p.name for p in target.glob("*.onnx"))
    print(f"  [ok  ] {show(target)}  onnx={onnx_files}")
    if not (target / "espeak-ng-data").is_dir():
        print("  [warn] không thấy espeak-ng-data/ - Piper cần thư mục này để phát âm")


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    """Reject archive members that would escape the destination directory."""
    base = path.resolve()
    for member in tar.getmembers():
        member_path = (base / member.name).resolve()
        if not str(member_path).startswith(str(base)):
            raise SystemExit(f"Archive chứa đường dẫn không an toàn: {member.name}")
    tar.extractall(path)  # noqa: S202 - members validated above


def fetch_wakeword() -> None:
    print("\n== Wake word: openWakeWord 'hey_jarvis' (+ melspectrogram/embedding/VAD)")
    try:
        import openwakeword.utils
    except ImportError:
        raise SystemExit(
            "Chưa cài openwakeword. Chạy: pip install -r requirements.txt"
        ) from None
    openwakeword.utils.download_models(["hey_jarvis"])
    import openwakeword

    onnx_path = Path(openwakeword.MODELS["hey_jarvis"]["model_path"].replace(".tflite", ".onnx"))
    if onnx_path.is_file():
        print(f"  [ok  ] {onnx_path.name} ({human(onnx_path.stat().st_size)})")
    else:
        print(f"  [warn] không thấy {onnx_path}")


# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tải model cho Jarvis")
    parser.add_argument(
        "--only",
        choices=["stt", "tts", "wakeword"],
        action="append",
        help="Chỉ tải phần này (có thể lặp lại). Mặc định tải tất cả.",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        choices=sorted(VOICES),
        help=f"Giọng TTS tiếng Việt (mặc định {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Tải thêm encoder/joiner fp32 (chậm hơn, chính xác hơn một chút)",
    )
    args = parser.parse_args(argv)

    targets = args.only or ["stt", "tts", "wakeword"]
    if "stt" in targets:
        fetch_stt(args.full_precision)
    if "tts" in targets:
        fetch_tts(args.voice)
    if "wakeword" in targets:
        fetch_wakeword()

    print("\nXong. Kiểm tra bằng: python -m jarvis doctor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
