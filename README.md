# Jarvis — trợ lý giọng nói tiếng Việt chạy local trên Windows

Jarvis là trợ lý giọng nói local-first cho Windows: wake word `Hey Jarvis` → nhận dạng tiếng Việt → LLM chạy trong LM Studio → trả lời bằng giọng nói. Mọi xử lý chính đều chạy trên máy: không có cloud fallback và không cần API key trả phí.

## Thành phần

- **STT offline:** [`hynt/Zipformer-30M-RNNT-6000h`](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) qua `sherpa-onnx`.
- **TTS offline:** VITS/Piper `vi_VN-vais1000-medium` qua `sherpa-onnx`.
- **Wake word:** cụm STT cục bộ `Xin chào` (hoặc model openWakeWord `hey_jarvis` khi cấu hình `stt_phrase: null`).
- **LLM local:** LM Studio + `nvidia/nemotron-3-nano-4b` Q4_K_M (~2,8 GB, vừa 4 GB VRAM). Jarvis gọi LM Studio qua REST OpenAI-compatible nên **không cần SDK riêng của nhà cung cấp**, và đổi sang server tương thích khác chỉ là đổi `llm.api_host`.
- **Intent router:** `jarvis/intents.py` xử lý thẳng các lệnh phổ biến (xem ngày giờ, mở app, phát bài hát, phím media) mà không gọi LLM.
- **Giao diện:** system tray có trạng thái idle / listening / thinking / speaking.
- **Tools:** ngày giờ, mở ứng dụng qua alias, phát bài hát theo tên, DuckDuckGo/SearxNG, media keys, lệnh Windows whitelist, và browser-use tùy chọn.

### Vì sao có intent router

`nemotron-3-nano-4b` là model *reasoning*: nó luôn sinh một khối `<think>` trước khi trả lời. Đo trên GTX 1650 4 GB:

| Đường đi | Ví dụ | Thời gian |
| --- | --- | --- |
| Intent router | `mở youtube`, `mấy giờ rồi`, `phát bài ...` | < 0,2 giây (`play_song` ~1,3 giây vì phải tìm video) |
| LLM + tool | `tìm thông tin về LM Studio` | ~35-40 giây |
| LLM thuần | `xin chào, bạn là ai` | ~15-30 giây |

Router cũng sửa một lỗi nội dung, không chỉ tốc độ: model suy luận bằng tiếng Anh nên cắt sai cụm từ tiếng Việt. Đo thực tế, `phát bài Em của ngày hôm qua` bị model gọi thành `play_song(song_name="Em")` vì nó hiểu "của ngày hôm qua" là "of yesterday". Router giữ nguyên văn tên bài.

## Yêu cầu

- Windows 10/11, Python **3.11** (dự án chỉ hỗ trợ `>=3.11,<3.12`).
- Microphone và loa hoạt động.
- [LM Studio](https://lmstudio.ai/) đã cài.
- Khuyến nghị 16 GB RAM. GPU 4 GB VRAM chạy được Nemotron 3 Nano 4B Q4_K_M; CUDA Toolkit 12.3 không tự động làm `sherpa-onnx` chạy CUDA.

## Cài đặt từ đầu

Mở PowerShell tại thư mục dự án:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Nếu PowerShell chặn activation trong phiên hiện tại, chạy `Set-ExecutionPolicy -Scope Process Bypass` rồi activation lại.

Tải các model offline (lệnh có thể chạy lại; file hợp lệ sẽ được bỏ qua):

```powershell
python scripts\download_models.py
```

Lệnh trên tải Zipformer vào `models\stt\zipformer-30m-rnnt-6000h`, giọng TTS vào `models\tts\vits-piper-vi_VN-vais1000-medium`, và cache wake word của openWakeWord. Muốn TTS nhẹ hơn dùng `python scripts\download_models.py --only tts --voice vivos`; sau đó cập nhật ba đường dẫn `tts` tương ứng trong `config.yaml`.

## LM Studio

Tải Nemotron 3 Nano 4B quantized trong LM Studio hoặc bằng CLI:

```powershell
& "$env:USERPROFILE\.lmstudio\bin\lms.exe" get "https://huggingface.co/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF" -y
& "$env:USERPROFILE\.lmstudio\bin\lms.exe" server start --port 1234 --bind 127.0.0.1
& "$env:USERPROFILE\.lmstudio\bin\lms.exe" load nvidia/nemotron-3-nano-4b --gpu max --context-length 8192 --yes
```

Hoặc trong LM Studio, mở **Developer**, bấm **Start Server** và tải model. Mặc định `config.yaml` dùng `localhost:1234` và model key `nvidia/nemotron-3-nano-4b`. Kiểm tra key cài thực tế bằng `lms ls`; nếu khác, sửa `llm.model` trong `config.yaml`. Không bật CORS hoặc bind `0.0.0.0` trừ khi bạn chủ động cần truy cập từ mạng khác.

Hai lưu ý riêng cho model reasoning này:

- Giữ `llm.max_tokens` ở mức rộng (mặc định 1024). Phần `<think>` cũng tiêu tokens, nên budget 512 có thể bị suy luận ăn hết và trả về câu rỗng.
- Phần suy luận **không bao giờ** được đọc ra loa. LM Studio trả nó ở field `reasoning_content` riêng, và `jarvis/llm.py` còn lọc thêm thẻ `<think>` để phòng trường hợp server cấu hình khác.

## Khởi chạy

```powershell
# Kiểm tra cấu hình, model, audio, LM Studio và tools
python -m jarvis doctor

# Chạy đầy đủ: tray + wake word
.\.venv\Scripts\python.exe -m jarvis run

# Thay wake word bằng Enter (dễ debug)
python -m jarvis run --push-to-talk

# Không dùng tray
python -m jarvis run --no-tray
```

Khi wake word được kích hoạt, Jarvis ghi âm đến khi phát hiện khoảng lặng, chuyển tiếng nói thành văn bản, gọi LLM local, sau đó đọc câu trả lời. Dùng `python -m jarvis devices` để xem hoặc chọn thiết bị audio; đặt `audio.input_device` / `audio.output_device` thành index hoặc một phần tên thiết bị trong `config.local.yaml`.

## Lệnh kiểm tra nhanh

```powershell
python -m jarvis --print-config
python -m jarvis chat "Xin chào" --no-tools
python -m jarvis chat "bây giờ mấy giờ rồi"        # intent router, không gọi LLM
python -m jarvis chat "phát bài Em của ngày hôm qua"
python -m jarvis speak "Xin chào, tôi là Jarvis" --out hello.wav
python -m jarvis transcribe hello.wav
python -m jarvis shell battery_status
python -m jarvis open notepad
python -m jarvis media volume_up
python -m jarvis search "thời tiết Hà Nội hôm nay"
python -m pytest -q
```

`chat` đi qua đúng đường mà một lượt nói đi qua: intent router trước, LLM sau. Dùng `--no-tools` để bỏ cả hai và nói trực tiếp với model. Nếu model gọi tool chưa ổn định, hãy nói ngắn và rõ ý định; các tool vẫn bị giới hạn bởi cấu hình và không bao giờ được tự sinh lệnh shell.

## Cấu hình

`config.yaml` là cấu hình dùng chung. Tạo `config.local.yaml` (đã được `.gitignore`) để ghi đè riêng cho máy, ví dụ:

```yaml
audio:
  input_device: "Microphone Array"
  output_device: "Speakers"
wake_word:
  threshold: 0.65
llm:
  model: nvidia/nemotron-3-nano-4b
```

Các đường dẫn tương đối được tính từ thư mục chứa file config. Xem toàn bộ cấu hình đã parse bằng `python -m jarvis --print-config`.

### CUDA và hiệu năng

Cấu hình mặc định dùng provider `cpu` cho STT/TTS; encoder/joiner Zipformer int8 vẫn chạy nhanh trong smoke test. CUDA Toolkit 12.3 riêng lẻ không đủ để đổi provider: cần cài bản `sherpa-onnx` có CUDA tương thích, sau đó mới đổi `stt.provider` và `tts.provider` sang `cuda`. Giữ CPU mặc định tránh lỗi DLL/runtime trên máy mới.

### Wake word

Cấu hình mặc định nhận câu đánh thức tiếng Việt `Xin chào` bằng STT cục bộ (`wake_word.stt_phrase`). Cách này không đòi model ONNX riêng, nhưng chậm hơn và kém chống nhiễu hơn wake-word model chuyên dụng vì Jarvis phải nhận dạng một câu nói hoàn chỉnh.

Muốn dùng openWakeWord, đặt `wake_word.stt_phrase: null` và giữ `wake_word.model: hey_jarvis`, hoặc cung cấp đường dẫn tới model `.onnx` tùy chỉnh. Với mode openWakeWord, nếu bị kích hoạt nhầm, tăng `wake_word.threshold` lên khoảng `0.6–0.7`; nếu khó bắt, giảm nhẹ xuống. Để debug mic trước, dùng `run --push-to-talk` thay vì tắt an toàn VAD.

Trong cuộc hội thoại, nói `Bái bai` để Jarvis quay lại chờ câu đánh thức. Cấu hình cũng chấp nhận alias STT `ba bai` khi model nhận câu nói không dấu hoặc biến âm.

## An toàn tools

- `tools.shell.whitelist` là danh sách duy nhất các lệnh được chạy. Jarvis xây dựng `argv` cố định, dùng `shell=False`, và không đưa câu lệnh tự do từ LLM vào CMD/PowerShell.
- Chỉ `open_folder` nhận một tham số đường dẫn, được kiểm tra bằng regex; PowerShell không nhận tham số do LLM cung cấp.
- `tools.apps.aliases` chỉ mở những alias do bạn khai báo. Thêm alias/lệnh mới vào `config.yaml` hoặc `config.local.yaml`, rồi kiểm tra bằng CLI trước khi dùng voice.
- Các lệnh có tác động như `lock_screen` chỉ nằm trong whitelist nếu bạn giữ chúng trong config. Hãy xem kỹ thay đổi config trước khi chạy.

## Browser-use (tùy chọn)

Browser automation bị tắt mặc định vì nó nặng, chậm hơn và có thể thao tác trang web theo yêu cầu. Nó vẫn dùng LM Studio local, không dùng cloud fallback.

```powershell
python -m pip install -r requirements-browser.txt
python -m playwright install chromium
```

Sau đó đặt `tools.browser_use.enabled: true` trong `config.local.yaml` và chạy:

```powershell
python -m jarvis browse "Mở YouTube và tìm nhạc lofi"
```

Chỉ giao tác vụ trên website/tài khoản mà bạn tin cậy. Khi bật browser-use, để `headless: false` ban đầu để quan sát thao tác; `use_vision: false` là hợp lý cho GPU 4 GB.

## Xử lý sự cố

| Triệu chứng | Cách xử lý |
| --- | --- |
| `LM Studio: LỖI` trong `doctor` | Bật Local Server tại `localhost:1234`, tải model và kiểm tra `lms ls`. |
| Jarvis im lặng hoặc trả lời rỗng | Model reasoning đôi khi sinh `<think></think>` rồi dừng. `llm.empty_reply_retries` (mặc định 2) tự thử lại; tăng `llm.max_tokens` nếu vẫn gặp. |
| Trả lời chậm 30 giây trở lên | Bình thường cho model reasoning trên GPU 4 GB. Các lệnh phổ biến đã được intent router xử lý dưới 1 giây; nạp model bằng `--gpu max` để đảm bảo không offload sang CPU. |
| Jarvis đọc cả đoạn suy luận tiếng Anh | Không nên xảy ra. Kiểm tra `llm.api_host` đang trỏ đúng LM Studio; `jarvis/llm.py` lọc `<think>` và chỉ đọc `content`. |
| Không tìm thấy model STT/TTS | Chạy lại `python scripts\download_models.py`, rồi `python -m jarvis doctor`. |
| Không có hoặc sai microphone | Chạy `python -m jarvis devices`; cập nhật `audio.input_device` trong `config.local.yaml`. |
| Wake word kích hoạt nhầm | Tăng `wake_word.threshold`; kiểm tra đúng microphone và môi trường ồn. |
| TTS/STT lỗi CUDA/DLL | Giữ provider `cpu`; chỉ đổi sang CUDA khi đã cài runtime Sherpa CUDA tương thích. |
| `browser-use chưa được cài` | Cài `requirements-browser.txt`, sau đó `python -m playwright install chromium`; bật cấu hình tool. |

## Phát triển và kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Bộ kiểm thử mock phần cứng/LLM và có smoke kiểm tra model audio khi asset đã tải. Không cần LM Studio để chạy unit test; chỉ cần nó cho `chat`, tool-calling và pipeline hội thoại đầy đủ.
