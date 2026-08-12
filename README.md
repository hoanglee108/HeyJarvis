# Jarvis — trợ lý giọng nói tiếng Việt chạy local trên Windows

Jarvis là trợ lý giọng nói local-first cho Windows: wake word `Hey Jarvis` → nhận dạng tiếng Việt → LLM chạy trong LM Studio → trả lời bằng giọng nói. Mọi xử lý chính đều chạy trên máy: không có cloud fallback và không cần API key trả phí.

## Thành phần

- **STT offline:** [`hynt/Zipformer-30M-RNNT-6000h`](https://huggingface.co/hynt/Zipformer-30M-RNNT-6000h) qua `sherpa-onnx`.
- **TTS offline:** VITS/Piper `vi_VN-vais1000-medium` qua `sherpa-onnx`.
- **Wake word:** openWakeWord `hey_jarvis`.
- **LLM local:** LM Studio + `qwen2.5-3b-instruct` Q4 (đã phù hợp máy có 4 GB VRAM).
- **Giao diện:** system tray có trạng thái idle / listening / thinking / speaking.
- **Tools:** lệnh Windows whitelist, mở ứng dụng qua alias, media keys, DuckDuckGo/SearxNG và browser-use tùy chọn.

## Yêu cầu

- Windows 10/11, Python **3.11** (dự án chỉ hỗ trợ `>=3.11,<3.12`).
- Microphone và loa hoạt động.
- [LM Studio](https://lmstudio.ai/) đã cài.
- Khuyến nghị 16 GB RAM. GPU 4 GB VRAM có thể dùng Qwen2.5 3B Q4; CUDA Toolkit 12.3 không tự động làm `sherpa-onnx` chạy CUDA.

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

Tải Qwen2.5 3B Instruct quantized trong LM Studio hoặc bằng CLI:

```powershell
& "$env:USERPROFILE\.lmstudio\bin\lms.exe" get "https://huggingface.co/lmstudio-community/Qwen2.5-3B-Instruct-GGUF" -y
& "$env:USERPROFILE\.lmstudio\bin\lms.exe" server start --port 1234 --bind 127.0.0.1
& "$env:USERPROFILE\.lmstudio\bin\lms.exe" load qwen2.5-3b-instruct --gpu max --context-length 4096 --yes
```

Hoặc trong LM Studio, mở **Developer**, bấm **Start Server** và tải model. Mặc định `config.yaml` dùng `localhost:1234` và model key `qwen2.5-3b-instruct`. Kiểm tra key cài thực tế bằng `lms ls`; nếu khác, sửa `llm.model` trong `config.yaml`. Không bật CORS hoặc bind `0.0.0.0` trừ khi bạn chủ động cần truy cập từ mạng khác.

## Khởi chạy

```powershell
# Kiểm tra cấu hình, model, audio, LM Studio và tools
python -m jarvis doctor

# Chạy đầy đủ: tray + wake word
python -m jarvis run

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
python -m jarvis speak "Xin chào, tôi là Jarvis" --out hello.wav
python -m jarvis transcribe hello.wav
python -m jarvis shell current_datetime
python -m jarvis open notepad
python -m jarvis media volume_up
python -m jarvis search "thời tiết Hà Nội hôm nay"
python -m pytest -q
```

`chat` có tools bật mặc định. Nếu model nhỏ gọi tool chưa ổn định, thử yêu cầu ngắn, rõ ý định; các tool vẫn bị giới hạn bởi cấu hình và không được tự sinh lệnh shell.

## Cấu hình

`config.yaml` là cấu hình dùng chung. Tạo `config.local.yaml` (đã được `.gitignore`) để ghi đè riêng cho máy, ví dụ:

```yaml
audio:
  input_device: "Microphone Array"
  output_device: "Speakers"
wake_word:
  threshold: 0.65
llm:
  model: qwen2.5-3b-instruct
```

Các đường dẫn tương đối được tính từ thư mục chứa file config. Xem toàn bộ cấu hình đã parse bằng `python -m jarvis --print-config`.

### CUDA và hiệu năng

Cấu hình mặc định dùng provider `cpu` cho STT/TTS; encoder/joiner Zipformer int8 vẫn chạy nhanh trong smoke test. CUDA Toolkit 12.3 riêng lẻ không đủ để đổi provider: cần cài bản `sherpa-onnx` có CUDA tương thích, sau đó mới đổi `stt.provider` và `tts.provider` sang `cuda`. Giữ CPU mặc định tránh lỗi DLL/runtime trên máy mới.

### Wake word

Model có sẵn là `hey_jarvis`. Nếu bị kích hoạt nhầm, tăng `wake_word.threshold` lên khoảng `0.6–0.7`; nếu khó bắt, giảm nhẹ xuống. Để debug mic trước, dùng `run --push-to-talk` thay vì tắt an toàn VAD.

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
