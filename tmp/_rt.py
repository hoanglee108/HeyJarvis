import logging
logging.disable(logging.INFO)
from jarvis.config import load_config
from jarvis.audio import Speaker, write_wav
from jarvis.tts import TextToSpeech
from jarvis.stt import SpeechToText
cfg = load_config("config.yaml")
tts = TextToSpeech(cfg.tts, Speaker(cfg.audio))
stt = SpeechToText(cfg.stt)
tests = [
  "Xin chào, hôm nay trời rất đẹp.",
  "Mở thư mục tải về giúp tôi.",
  "Bây giờ là mấy giờ rồi.",
  "Hôm nay tôi muốn nghe nhạc.",
  "Tăng âm lượng lên một chút.",
  "Thời tiết Hà Nội hôm nay thế nào.",
]
for s in tests:
    a = tts.synthesize(s)
    p = write_wav("tmp/rt.wav", a.samples, a.sample_rate)
    print(repr(s), "->", repr(stt.transcribe_file(p)))
