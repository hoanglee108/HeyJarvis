# Jarvis — Local-First Personal Voice Agent

> **Tài liệu tầm nhìn ban đầu.** Một số quyết định kỹ thuật ở đây đã thay đổi sau khi
> triển khai thực tế: LLM hiện là `nvidia/nemotron-3-nano-4b` (không phải Qwen2.5-3B),
> và Jarvis gọi LM Studio qua REST OpenAI-compatible thay vì SDK `lmstudio-python`.
> Xem [`structure.md`](structure.md) để biết kiến trúc đang chạy và lý do thay đổi.

## 1. Vision

Xây dựng một trợ lý cá nhân kiểu **"Jarvis"** chạy **100% local trên Windows**, ưu tiên:

* Privacy / offline-first
* Chi phí vận hành gần như bằng 0
* Voice-first interaction
* Tool calling có kiểm soát
* Windows automation
* Web search
* Personal memory
* Kiến trúc modular, dễ mở rộng
* Có giá trị làm **GitHub showcase project**

Mục tiêu không phải tạo chatbot voice đơn thuần, mà xây dựng một **local AI agent có khả năng nghe → suy luận → sử dụng tool → thực thi → phản hồi bằng giọng nói**.

---

# 2. Target Hardware

## Current machine

* OS: Windows 11
* CPU: laptop CPU
* RAM: 16GB
* GPU: NVIDIA GTX 1650 Ti
* VRAM: 4GB
* Storage: SSD 512GB+
* CUDA capable

## Hardware constraints

Đây là constraint quan trọng nhất của MVP.

Không nên thiết kế architecture phụ thuộc vào:

* 7B+ model chạy full GPU
* Vision LLM nặng
* Multiple LLM instances chạy đồng thời
* Cloud API
* GPU-heavy browser automation
* Large vector database
* Multi-agent orchestration

### Model strategy

Ưu tiên model nhỏ, quantized:

1. Qwen2.5-3B-Instruct — primary candidate
2. Llama-3.2-3B-Instruct — alternative
3. Các model ~3B tương đương — có thể benchmark thêm

Target:

```text
3B model
Q4 quantization
~2-2.5GB VRAM
```

Mục tiêu là giữ đủ VRAM cho:

* OS / GPU overhead
* LLM inference
* voice pipeline nếu cần
* các tác vụ khác

Model phải được cấu hình qua config, KHÔNG hardcode.

Ví dụ:

```yaml
llm:
  provider: lmstudio
  model: qwen2.5-3b-instruct
  temperature: 0.2
```

---

# 3. Core Principles

## 3.1 Local-first

Mặc định:

```text
Microphone
    ↓
Local STT
    ↓
Local LLM
    ↓
Local Tools
    ↓
Local TTS
```

Không yêu cầu cloud API để hoạt động.

Web search là ngoại lệ vì bản chất cần Internet:

```text
Jarvis
    ↓
Web Search
    ↓
Internet
```

Nhưng không sử dụng API trả phí.

---

## 3.2 Security-first

LLM KHÔNG được phép tự sinh và chạy shell command tùy ý.

Sai:

```text
LLM
 ↓
os.system(llm_generated_command)
```

Đúng:

```text
LLM
 ↓
Tool Calling
 ↓
Permission Layer
 ↓
Whitelisted Tool
 ↓
Execution
```

Ví dụ:

```python
run_backend()
open_app("spotify")
git_status()
search_web("...")
```

thay vì:

```python
run_command("whatever the LLM generated")
```

---

# 4. High-Level Architecture

```text
                         ┌─────────────────────┐
                         │     WINDOWS UI      │
                         │   System Tray App   │
                         └──────────┬──────────┘
                                    │
                                    │ status
                                    ▼
┌──────────┐
│ Microphone│
└─────┬────┘
      │
      ▼
┌──────────────────┐
│  Wake Word       │
│  openWakeWord    │
│ "Hey Jarvis"     │
└────────┬─────────┘
         │ detected
         ▼
┌──────────────────┐
│ VAD / Recording  │
│ Silero VAD       │
└────────┬─────────┘
         │ audio
         ▼
┌──────────────────────────┐
│ Local STT                │
│ sherpa-onnx              │
│ Zipformer-30M-RNNT       │
│ Vietnamese               │
└────────┬─────────────────┘
         │ text
         ▼
┌──────────────────────────┐
│       JARVIS CORE        │
│                          │
│ Context                  │
│ Agent State              │
│ Tool Calling             │
│ Permission               │
│ Memory                   │
└────────┬─────────────────┘
         │
         │ lmstudio-python
         ▼
┌──────────────────────────┐
│       LM STUDIO          │
│                          │
│ Local LLM                │
│ Qwen 2.5 3B              │
└────────┬─────────────────┘
         │
         │ tool calls
         ▼
┌──────────────────────────────────────────┐
│                 TOOLS                    │
│                                          │
│  Shell       App       Web       Git     │
│  Tool        Tool      Search    Tool    │
│                                          │
│  PyAutoGUI       Browser-use (later)     │
└─────────────────────┬────────────────────┘
                      │
                      ▼
              Tool execution result
                      │
                      ▼
                 LM Studio
                      │
                      ▼
              Final response
                      │
                      ▼
┌──────────────────────────┐
│ Local TTS                │
│ sherpa-onnx / Piper      │
│ Vietnamese voice         │
└────────────┬─────────────┘
             │
             ▼
          Speaker
```

---

# 5. Voice Pipeline

## 5.1 Always-listening architecture

Jarvis does NOT continuously send audio to an LLM.

Instead:

```text
Microphone
    ↓
openWakeWord
    ↓
"Hey Jarvis"
    ↓
Start recording
    ↓
VAD
    ↓
Stop after silence
    ↓
STT
```

This dramatically reduces resource usage.

---

# 6. Wake Word

## Technology

Use:

```text
openWakeWord
```

Target wake word:

```text
Hey Jarvis
```

Reasons:

* Local
* Lightweight
* ONNX Runtime
* Windows compatible
* Designed for continuous listening
* Pretrained wake-word support
* CPU-friendly
* No API cost

Wake-word detection should remain active while the rest of the system is idle.

---

# 7. VAD / Recording

Use:

```text
Silero VAD
```

Purpose:

* Detect speech start/end
* Avoid fixed recording duration
* Reduce STT workload
* Improve perceived responsiveness

Pipeline:

```text
Wake word
    ↓
VAD starts
    ↓
User speaks
    ↓
Silence detected
    ↓
Stop recording
```

---

# 8. Vietnamese STT

## Primary stack

```text
sherpa-onnx
+
hynt/Zipformer-30M-RNNT-6000h
```

This is preferred over Whisper for MVP because:

* Very small model
* Designed for streaming ASR
* ONNX runtime
* CPU-friendly
* Offline
* Vietnamese support
* Low resource consumption
* Good fit for 16GB RAM / 4GB VRAM machine

Reference implementation:

```text
sherpa-vietnamese-asr
```

The STT model must run independently from LM Studio.

Important:

```text
LM Studio = LLM
sherpa-onnx = ASR/TTS runtime
```

Do not try to route ASR through LM Studio.

---

# 9. LLM

## Runtime

Use:

```text
LM Studio
```

Integration:

```text
lmstudio-python
```

Preferred API:

```python
model.act(...)
```

Use LM Studio's tool-calling capability instead of manually implementing the entire tool loop during MVP.

Concept:

```text
User text
    ↓
model.act()
    ↓
LLM decides:
    ├── no tool
    ├── web_search()
    ├── open_app()
    ├── git_status()
    └── run_whitelisted_command()
    ↓
Tool result
    ↓
LLM
    ↓
Final response
```

---

# 10. LLM Model

Primary:

```text
Qwen2.5-3B-Instruct
```

Alternative:

```text
Llama-3.2-3B-Instruct
```

Quantization:

```text
Q4
```

Target:

```text
~2GB model memory
```

Why 3B:

* Better fit for 4GB VRAM
* Faster than 7B
* Lower RAM pressure
* Faster tool calling
* Better latency for voice interaction
* Leaves resources for other processes

7B models can be experimented with later but are NOT the MVP baseline.

---

# 11. Tool Calling Architecture

Tools must be explicit and typed.

Example:

```python
tools = [
    web_search,
    open_app,
    git_status,
    run_allowed_command,
    media_control,
]
```

LLM chooses from tools.

It cannot directly execute arbitrary Python or shell.

---

# 12. Tool Categories

## 12.1 Shell Tool

Only predefined commands are allowed.

Example:

```text
git_status
git_pull
run_backend
run_frontend
docker_up
docker_down
npm_install
python_run_tests
```

Internally:

```text
Tool name
    ↓
Predefined command
    ↓
Validation
    ↓
subprocess
```

Never:

```text
LLM → arbitrary command string → subprocess
```

---

# 13. App Launcher

Use a user-defined alias map.

Example:

```yaml
apps:
  music: "C:/Apps/Spotify/Spotify.exe"
  vscode: "C:/Users/.../Code.exe"
  browser: "C:/Program Files/Google/Chrome/Application/chrome.exe"
```

User commands:

```text
"Jarvis, mở VS Code"

"Jarvis, mở nhạc"

"Jarvis, mở Facebook"
```

LLM maps natural language to:

```text
open_app("vscode")
open_app("music")
open_app("browser")
```

No need for computer vision or LLM GUI reasoning.

---

# 14. Desktop Automation

Use:

```text
PyAutoGUI
```

for deterministic tasks:

* Mouse
* Keyboard
* Media keys
* Fixed UI actions
* Simple application interaction

Example:

```text
play music
pause music
next song
volume up
volume down
```

PyAutoGUI should be preferred over an agent when the action is deterministic.

---

# 15. Browser Automation

## Not MVP-critical

Use:

```text
browser-use
```

only for tasks that cannot be represented as deterministic tools.

Example:

```text
"Jarvis, vào website X và tìm thông tin Y."

"Jarvis, mở trang quản lý và kiểm tra trạng thái deployment."
```

Architecture:

```text
Jarvis LLM
    ↓
browser_use()
    ↓
Browser-use Agent
    ↓
Local LM Studio
    ↓
Browser
```

Important constraint:

browser-use introduces another LLM reasoning loop.

Therefore:

```text
Simple task
→ PyAutoGUI / direct HTTP / Playwright

Complex web task
→ browser-use
```

Do NOT use browser-use for every browser action.

---

# 16. Web Search

Must remain free.

Preferred options:

```text
DuckDuckGo HTML
```

or:

```text
Self-hosted SearXNG
```

MVP:

```text
web_search(query)
```

Pipeline:

```text
User
 ↓
LLM
 ↓
web_search()
 ↓
DuckDuckGo / SearXNG
 ↓
Parse results
 ↓
LLM summarizes
 ↓
TTS
```

No paid API key.

---

# 17. Memory

## MVP

Use:

```text
SQLite
```

Store:

* Conversation history
* User preferences
* App aliases
* Tool configuration
* Recent tasks
* Jarvis settings

Do NOT immediately introduce a vector database.

---

## Future

Add:

```text
SQLite
+
Chroma / lightweight vector DB
```

for semantic memory.

Only add this after basic memory is proven useful.

---

# 18. Context Management

Because 3B models have limited reasoning capability, context must remain small.

Use:

```text
System prompt
+
Current user command
+
Relevant tool result
+
Short conversation history
+
Relevant memory
```

Avoid sending the entire conversation every time.

---

# 19. System Tray

Use a Windows system tray application.

Suggested states:

```text
IDLE
LISTENING
PROCESSING
EXECUTING_TOOL
SPEAKING
ERROR
```

Example:

```text
🟢 Idle

🔵 Listening...

🟡 Thinking...

🟠 Executing...

🟣 Speaking...

🔴 Error
```

The tray application is the primary UI for MVP.

No need to build a full desktop UI initially.

---

# 20. State Machine

Jarvis should explicitly maintain state.

```text
                    ┌────────────┐
                    │    IDLE    │
                    └─────┬──────┘
                          │
                    wake word
                          │
                          ▼
                  ┌───────────────┐
                  │   LISTENING   │
                  └───────┬───────┘
                          │
                    speech ended
                          │
                          ▼
                  ┌───────────────┐
                  │  PROCESSING   │
                  └───────┬───────┘
                          │
                     tool call
                          │
                          ▼
                  ┌───────────────┐
                  │ TOOL EXECUTION │
                  └───────┬───────┘
                          │
                          ▼
                  ┌───────────────┐
                  │   SPEAKING    │
                  └───────┬───────┘
                          │
                          ▼
                       IDLE
```

This makes system behavior easier to debug and display in the tray.

---

# 21. TTS

Use:

```text
sherpa-onnx
```

or:

```text
Piper
```

Vietnamese voice candidates:

```text
vivos
vais1000
25hours_single
```

MVP preference:

```text
vivos
```

Reason:

* Local
* Lightweight
* Vietnamese
* Easy deployment
* Low resource requirement

Trade-off:

Voice quality will not match cloud TTS.

This is acceptable for MVP because local-first is a core requirement.

---

# 22. Latency Target

Voice UX is more important than raw model intelligence.

Target approximate pipeline:

```text
Wake word detection
      ↓
< 100 ms
      ↓
Record
      ↓
STT
      ↓
~0.3-1s
      ↓
LLM
      ↓
~1-3s
      ↓
TTS
      ↓
~0.5-2s
```

Goal:

```text
Short command → audible response within ~2-5 seconds
```

Avoid unnecessary LLM calls.

---

# 23. Barge-in / Interrupt

Important future feature.

While Jarvis is speaking:

```text
TTS
 ↓
wake word detector still active
```

If user says:

```text
"Hey Jarvis"
```

then:

```text
cancel TTS
cancel current response if possible
return to LISTENING
```

This creates a much more natural Jarvis-like UX.

Not required for first MVP but architecture should not prevent it.

---

# 24. Recommended Python Project Structure

```text
jarvis/
│
├── app/
│   ├── main.py
│   └── state.py
│
├── voice/
│   ├── wakeword.py
│   ├── vad.py
│   ├── stt.py
│   ├── tts.py
│   └── audio.py
│
├── agent/
│   ├── orchestrator.py
│   ├── prompts.py
│   ├── context.py
│   └── model.py
│
├── tools/
│   ├── registry.py
│   ├── shell.py
│   ├── apps.py
│   ├── web.py
│   ├── git.py
│   ├── media.py
│   ├── desktop.py
│   └── browser.py
│
├── security/
│   ├── whitelist.py
│   └── permissions.py
│
├── memory/
│   ├── sqlite.py
│   └── manager.py
│
├── ui/
│   └── tray.py
│
├── config/
│   ├── config.yaml
│   ├── apps.yaml
│   └── commands.yaml
│
├── models/
│   └── README.md
│
├── tests/
│
├── requirements.txt
├── README.md
└── main.py
```

---

# 25. MVP Scope

## MVP-0 — Voice Loop

Implement:

```text
Mic
 ↓
Wake word
 ↓
STT
 ↓
LLM
 ↓
TTS
```

Example:

```text
User:
"Hey Jarvis"

Jarvis:
"Yes?"

User:
"Xin chào Jarvis"

Jarvis:
"Xin chào. Tôi có thể giúp gì?"
```

---

## MVP-1 — Tools

Add:

```text
open_app()
web_search()
git_status()
run_allowed_command()
media_control()
```

Example:

```text
"Hey Jarvis, mở VS Code"

"Hey Jarvis, kiểm tra git status"

"Hey Jarvis, tìm thông tin về FastAPI"

"Hey Jarvis, chạy backend"
```

---

## MVP-2 — Windows Automation

Add:

```text
PyAutoGUI
PowerShell
CMD
Docker
Git
App aliases
```

Example:

```text
"Jarvis, chạy Docker backend"

"Jarvis, mở Spotify và play nhạc"

"Jarvis, mở project AI"
```

---

## MVP-3 — Memory

Add:

```text
SQLite
```

Examples:

```text
"Jarvis, nhớ rằng project backend nằm ở D:/Projects/backend"

"Jarvis, project backend của tôi ở đâu?"
```

---

## MVP-4 — Web Agent

Only now add:

```text
browser-use
```

for complex web tasks.

---

# 26. Features Explicitly Deferred

Do NOT implement these in MVP:

```text
❌ Cloud LLM fallback
❌ Cloud STT
❌ Cloud TTS
❌ 7B+ default model
❌ Multi-agent
❌ Vision LLM
❌ RAG
❌ Large vector database
❌ Autonomous unrestricted shell
❌ Full desktop computer-use agent
❌ Dockerized desktop application
❌ Complex React frontend
❌ Always-on browser-use
```

These can be future milestones.

---

# 27. Recommended Dependency Stack

```text
Core
├── Python 3.11
├── LM Studio
└── lmstudio-python

Voice
├── openwakeword
├── sherpa-onnx
├── Silero VAD
└── Piper / sherpa-onnx TTS

Windows
├── pyautogui
├── pywin32
├── subprocess
└── pystray

Web
├── requests
├── BeautifulSoup
└── DuckDuckGo / SearXNG

Automation
├── Playwright
└── browser-use (later)

Memory
└── SQLite

Packaging
└── PyInstaller
```

---

# 28. Configuration Strategy

Everything resource-intensive must be configurable.

```yaml
system:
  os: windows

llm:
  provider: lmstudio
  model: qwen2.5-3b-instruct
  temperature: 0.2

voice:
  wake_word: hey_jarvis

stt:
  runtime: sherpa-onnx
  model: Zipformer-30M-RNNT-6000h
  language: vi

tts:
  runtime: sherpa-onnx
  voice: vivos

web:
  provider: duckduckgo

automation:
  browser_use: false

security:
  shell_mode: whitelist
```

---

# 29. Security Model

Three levels:

```text
SAFE
  ↓
No confirmation required

CONFIRM
  ↓
Ask user before execution

BLOCKED
  ↓
Never allowed
```

Example:

```text
SAFE:
open_app()
git_status()
web_search()

CONFIRM:
docker_down()
git_pull()
delete_file()

BLOCKED:
format_disk()
shutdown_system()
arbitrary_shell()
credential_access()
```

Future:

```text
Tool
 ↓
Risk score
 ↓
Permission policy
 ↓
Execute / Ask / Block
```

---

# 30. Performance Strategy

The biggest performance bottleneck is the GPU/LLM.

Therefore:

### CPU

Use CPU for:

```text
Wake word
VAD
STT
TTS
simple automation
```

### GPU

Reserve GPU primarily for:

```text
Local LLM
```

Avoid running multiple GPU-heavy models simultaneously.

---

# 31. Reference Projects

The project should take inspiration from multiple existing open-source projects instead of forking one.

## Voice loop

```text
Atzingen/hey-jarvis
```

Useful concepts:

* Wake word
* Recording lifecycle
* TTS loop
* Interrupt / barge-in
* Voice assistant state

## Vietnamese ASR

```text
welcomyou/sherpa-vietnamese-asr
```

Useful concepts:

* sherpa-onnx
* Zipformer
* Vietnamese offline ASR

## Agent architecture

```text
Leon AI Assistant
```

Useful concepts:

* Skills
* Tools
* Agent architecture
* Memory
* Modular assistant design

## Voice UX

```text
openjarvis
```

Useful concepts:

* Wake word
* Conversation window
* Voice interaction
* Local voice architecture

These projects are references only.

Jarvis should implement its own simplified architecture optimized for:

```text
Windows
16GB RAM
GTX 1650Ti 4GB
Vietnamese
LM Studio
Local-first
```

---

# 32. Core Design Decision

The project is NOT:

```text
Voice chatbot
```

It is:

```text
Voice Interface
       +
Local LLM
       +
Typed Tool Calling
       +
Windows Automation
       +
Memory
       +
Web Search
```

The LLM is the **brain**, but it does not directly control the computer.

```text
                 JARVIS
                    │
          ┌─────────┴─────────┐
          │                   │
        Brain              Hands
          │                   │
      LM Studio            Tools
          │                   │
      Reasoning       ┌───────┼────────┐
                      │       │        │
                    Shell   Windows   Web
```

This separation is the core architecture.

---

# 33. GitHub Showcase Goals

The repository should demonstrate:

* Local LLM integration
* Voice AI
* Vietnamese ASR
* Tool calling
* Agent architecture
* Windows automation
* Security / permission layer
* Web search
* Memory
* Async programming
* System tray application
* Modular software architecture

README should include:

```text
Architecture diagram
↓
Demo GIF/video
↓
Voice interaction demo
↓
Tool calling demo
↓
Security model
↓
Performance benchmark
↓
Installation
↓
Configuration
↓
Roadmap
```

The project should emphasize:

> **A privacy-first Vietnamese local AI agent for Windows that can hear, reason, search, and safely operate the user's computer.**

---

# 34. Final MVP Architecture

```text
                   ┌───────────────┐
                   │   Microphone  │
                   └───────┬───────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ openWakeWord   │
                  └───────┬────────┘
                          │
                    Hey Jarvis
                          │
                          ▼
                  ┌────────────────┐
                  │   Silero VAD   │
                  └───────┬────────┘
                          │
                          ▼
                  ┌────────────────┐
                  │ sherpa-onnx    │
                  │ Zipformer 30M  │
                  └───────┬────────┘
                          │
                       Vietnamese
                          │
                          ▼
                  ┌────────────────┐
                  │  LM Studio     │
                  │  Qwen 3B Q4    │
                  └───────┬────────┘
                          │
                     Tool Calling
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
        App Launcher   Web Search   Safe Shell
             │            │            │
             └────────────┼────────────┘
                          │
                          ▼
                  ┌────────────────┐
                  │ Final Response │
                  └───────┬────────┘
                          │
                          ▼
                  ┌────────────────┐
                  │ sherpa/Piper   │
                  │ Vietnamese TTS │
                  └───────┬────────┘
                          │
                          ▼
                       Speaker
```

## MVP success criteria

Jarvis can reliably handle:

```text
"Hey Jarvis, mở VS Code"

"Hey Jarvis, mở Spotify"

"Hey Jarvis, kiểm tra git status"

"Hey Jarvis, chạy backend"

"Hey Jarvis, tìm thông tin về LM Studio"

"Hey Jarvis, tăng âm lượng"

"Hey Jarvis, nhớ project backend nằm ở đâu"

"Hey Jarvis, project backend của tôi ở đâu?"
```

All of the above must work:

```text
100% local
+
Vietnamese
+
Windows
+
GTX 1650Ti 4GB
+
16GB RAM
```

before adding complex browser agents, RAG, vision, multi-agent systems, or cloud services.
