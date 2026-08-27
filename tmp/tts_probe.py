"""Temporary probe: is the tail of a long spoken reply actually synthesized?"""

from __future__ import annotations

from jarvis.audio import Speaker
from jarvis.config import load_config
from jarvis.stt import SpeechToText
from jarvis.tts import TextToSpeech, clean_for_speech

REPLY = (
    "Bạn có thể xem thông tin thời tiết hàng giờ tại Thành phố Hồ Chí Minh trên các trang web sau:\n"
    "- [Thời tiết hàng giờ ở Thành phố Hồ Chí Minh]"
    "(https://www.accuweather.com/vi/vn/ho-chi-minh-city/353981/hourly-weather-forecast/353981)\n"
    "- [Dự báo thời tiết hàng giờ tại Hồ Chí Minh](https://thoitiet.vn/ho-chi-minh/theo-gio)\n"
    "Bạn có thể chọn một trang web để xem thông tin thời tiết cụ thể."
)


def main() -> None:
    config = load_config("config.yaml")
    tts = TextToSpeech(config.tts, Speaker(config.audio))
    stt = SpeechToText(config.stt)

    spoken = clean_for_speech(REPLY)
    print(f"spoken_chars={len(spoken)}")

    result = tts.synthesize(REPLY)
    print(f"audio={result.duration:.2f}s")

    tail_seconds = 6.0
    tail = result.samples[-int(tail_seconds * result.sample_rate) :]
    print(f"tail_transcript={stt.transcribe(tail, result.sample_rate)!r}")
    print(f"full_transcript={stt.transcribe(result.samples, result.sample_rate)!r}")


if __name__ == "__main__":
    main()
