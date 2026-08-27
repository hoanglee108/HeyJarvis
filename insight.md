# insight.md

> **Ghi chép research ban đầu.** Hai kết luận trong tài liệu này đã bị thực tế sửa lại:
> model LLM là `nvidia/nemotron-3-nano-4b`, và Jarvis không dùng SDK `lmstudio-python`
> mà gọi REST OpenAI-compatible để tách được phần suy luận của model reasoning.
> Xem [`structure.md`](structure.md) để biết trạng thái hiện tại.

## Problem Statement
Xây dựng một trợ lý giọng nói cá nhân kiểu "Jarvis" chạy local-first trên Windows,
dùng cho công việc cá nhân và làm project showcase GitHub. Agent nghe lệnh tiếng Việt
qua wake word, xử lý bằng LLM local, và trả lời bằng giọng nói. Agent có thể thao tác
máy (lệnh cmd whitelist, mở app, điều khiển UI/web) và tra cứu thông tin qua web search.

## Requirements (đã chốt qua Q&A)
- OS: Windows only.
- Hardware: 16GB RAM, GTX 1650Ti 4GB VRAM. LLM full local qua LM Studio, KHÔNG fallback cloud.
- Voice pipeline: always-listening mic ở chế độ chờ → wake word ("Hey Jarvis") kích hoạt →
  STT → LLM → TTS trả lời bằng giọng nói.
- STT: model `hynt/Zipformer-30M-RNNT-6000h` (tiếng Việt), chạy qua sherpa-onnx (KHÔNG qua
  LM Studio — LM Studio không hỗ trợ ASR).
- LLM: LM Studio (local), dùng lmstudio-python SDK để tool-calling.
- TTS: sherpa-onnx / Piper voice tiếng Việt (local), full offline.
- Web search: nguồn free/scraping (DuckDuckGo HTML hoặc SearxNG self-host), không dùng API
  key trả phí.
- Thao tác máy: chỉ chạy lệnh cmd/PowerShell được whitelist trước (an toàn, không cho LLM
  tự sinh lệnh tùy ý).
- Mở app đã cài (nhạc, Facebook...): qua alias map do người dùng định nghĩa.
- Tương tác UI/web nâng cao: dùng browser-use (web automation agent) + PyAutoGUI (desktop
  mouse/keyboard, media keys).
- Interface giám sát: system tray icon (Windows) hiển thị trạng thái
  idle / nghe / xử lý / đang nói.

## Research Findings
- **openWakeWord** (pip install openwakeword): dùng onnxruntime, chạy tốt trên Windows,
  có sẵn pretrained model "hey jarvis", nhẹ, phù hợp chạy liên tục 24/7 mà không tốn
  nhiều CPU/GPU.
- **sherpa-onnx** (pip install sherpa-onnx): runtime đúng cho cả STT (Zipformer RNNT) và
  TTS (VITS). Hỗ trợ CPU hoặc CUDA. Đây là runtime độc lập với LM Studio.
- **TTS tiếng Việt local**: Piper/sherpa-onnx có voice `vivos`, `vais1000`,
  `25hours_single` (tiếng Việt, chất lượng tăng dần theo thứ tự, `vivos` là lựa chọn
  cân bằng chất lượng/tài nguyên tốt cho MVP). Giọng máy, không tự nhiên như cloud TTS
  — đây là trade-off đã chấp nhận để giữ local-first.
- **LM Studio tool-calling**: SDK `lmstudio-python` cung cấp `model.act(prompt, [tool_fns])`
  tự xử lý toàn bộ loop gọi tool, không cần tự viết orchestration. Đây là cách tích hợp
  gọn nhất so với raw OpenAI-compatible endpoint.
- **browser-use**: không phải "tool" đơn giản mà là agent có LLM loop riêng (đọc DOM/
  screenshot, quyết định click/gõ). Cần cấp cho nó một LLM — có thể point về cùng LM
  Studio instance qua `ChatOpenAI(base_url="http://localhost:1234/v1")` để giữ
  local-first, nhưng sẽ tạo thêm tầng gọi LLM (tăng latency, đặc biệt nếu dùng screenshot
  vision trên model nhỏ 3B chạy trên 4GB VRAM).
- **PyAutoGUI**: đơn giản, không cần LLM riêng, phù hợp cho media keys, click theo tọa độ
  cố định, gõ text — rủi ro thấp hơn browser-use vì không có reasoning loop riêng.
- **Model LLM cho 4GB VRAM**: khuyến nghị bắt đầu với **Qwen2.5-3B-Instruct** hoặc
  **Llama-3.2-3B-Instruct** (quantized Q4_K_M, ~2GB) — đủ nhỏ để offload full lên GPU,
  tool-calling ổn định. Có thể thử **Qwen2.5-7B-Instruct Q4** (~4.5GB) nếu chấp nhận
  offload phần layer lên CPU/RAM (chậm hơn). Nên để cấu hình model là tham số dễ đổi
  trong config, không hardcode.

## Architecture

```mermaid
flowchart TD
    Mic[Microphone - always on] --> WW[Wake Word Detector<br/>openWakeWord]
    WW -- "hey jarvis" detected --> Rec[Record + VAD<br/>cắt khi im lặng]
    Rec --> STT[STT<br/>sherpa-onnx + Zipformer-30M]
    STT --> Orchestrator[LLM Orchestrator<br/>LM Studio via lmstudio-python act]
    Orchestrator -- tool call --> Shell[Whitelisted CMD/PowerShell Tool]
    Orchestrator -- tool call --> AppLauncher[App Alias Launcher<br/>+ PyAutoGUI media keys]
    Orchestrator -- tool call --> WebSearch[Web Search Tool<br/>DuckDuckGo/SearxNG]
    Orchestrator -- tool call --> BrowserUse[browser-use Agent<br/>local LLM driven]
    Shell --> Orchestrator
    AppLauncher --> Orchestrator
    WebSearch --> Orchestrator
    BrowserUse --> Orchestrator
    Orchestrator --> TTS[TTS<br/>sherpa-onnx VITS vi_VN]
    TTS --> Speaker[Speaker Output]
    WW -.status.-> Tray[System Tray Icon<br/>idle/listening/thinking/speaking]
    Rec -.status.-> Tray
    Orchestrator -.status.-> Tray
    TTS -.status.-> Tray
