# mvp.md

## Task Breakdown

Task 1: Project scaffolding & config system
- Objective: Khởi tạo cấu trúc project Python (venv, dependency management), và hệ thống
  config (YAML/JSON) chứa: đường dẫn model STT/TTS, danh sách lệnh whitelist, alias map
  ứng dụng, LM Studio endpoint/model name.
- Guidance: Dùng `pydantic` hoặc `dataclasses` để validate config khi load. Tách rõ
  module `config.py` khỏi logic runtime.
- Test: Unit test load config hợp lệ, và test reject config thiếu field bắt buộc.
- Demo: Chạy `python -m jarvis --print-config` in ra config đã parse.

Task 2: LM Studio connectivity layer
- Objective: Wrapper kết nối tới LM Studio (qua `lmstudio-python` SDK) để gửi prompt text
  và nhận response, chưa có tool-calling.
- Guidance: Bọc SDK trong 1 class `LLMClient` để dễ test/mock, xử lý lỗi khi LM Studio
  offline (raise exception rõ nghĩa).
- Test: Unit test với LM Studio client mock; smoke test thủ công gọi LM Studio thật.
- Demo: `jarvis chat "Xin chào"` in ra câu trả lời text từ LLM local.

Task 3: STT integration (sherpa-onnx + Zipformer)
- Objective: Chuyển file WAV tiếng Việt thành text bằng model Zipformer-30M-RNNT-6000h
  qua sherpa-onnx.
- Guidance: Viết module `stt.py` load model 1 lần (session reuse), expose hàm
  `transcribe(audio) -> str`.
- Test: Unit test với file WAV mẫu tiếng Việt, assert transcript không rỗng và chứa từ
  khóa kỳ vọng.
- Demo: `jarvis transcribe sample.wav` in ra văn bản tiếng Việt.

Task 4: TTS integration (sherpa-onnx VITS vi_VN)
- Objective: Chuyển text thành audio và phát ra loa bằng voice tiếng Việt local
  (vivos hoặc tương đương).
- Guidance: Module `tts.py` với hàm `speak(text)`, dùng `sounddevice`/`simpleaudio` để
  phát audio.
- Test: Unit test generate file audio, assert file không rỗng và có độ dài hợp lý.
- Demo: `jarvis speak "Xin chào, tôi là Jarvis"` phát ra giọng nói.

Task 5: Wire pipeline end-to-end (push-to-talk tạm thời)
- Objective: Ghép STT → LLM → TTS thành 1 vòng lặp, kích hoạt bằng Enter (chưa có wake
  word), để xác nhận toàn bộ chuỗi hoạt động.
- Guidance: Ghi âm N giây khi nhấn Enter, transcribe, gửi cho LLM (chưa tool), đọc kết
  quả. Tổ chức thành `pipeline.py` orchestration.
- Test: Integration test dùng WAV cố định thay mic thật, assert pipeline chạy hết không
  lỗi và sinh ra audio output.
- Demo: Nói vào mic sau khi nhấn Enter, nghe Jarvis trả lời bằng giọng nói — vòng lặp
  hội thoại đầy đủ (chưa hands-free).

Task 6: Wake word detection + auto end-of-speech
- Objective: Thay trigger Enter bằng openWakeWord ("hey jarvis") chạy liên tục trên mic,
  và tự động cắt ghi âm khi phát hiện im lặng (VAD).
- Guidance: Chạy wake-word detector trong loop riêng (thread), khi trigger thì bắt đầu
  ghi âm với silence-detection cutoff (energy-based hoặc `webrtcvad`).
- Test: Unit test feed audio clip chứa "hey jarvis" và assert event fires; unit test cho
  logic cắt khi im lặng.
- Demo: Nói "Hey Jarvis", agent tự động ghi âm, xử lý, và trả lời — vòng lặp hands-free
  hoàn chỉnh, không cần bấm phím.

Task 7: System tray icon với trạng thái
- Objective: Icon tray Windows hiển thị trạng thái idle / nghe / đang xử lý / đang nói.
- Guidance: Dùng `pystray` + state machine đơn giản (`enum State`), cập nhật icon theo
  sự kiện từ pipeline.
- Test: Unit test cho state machine transitions (idle→listening→thinking→speaking→idle).
- Demo: Icon tray đổi trạng thái theo thời gian thực trong lúc thực hiện 1 lượt hội thoại
  bằng giọng nói.

Task 8: Whitelisted shell/cmd tool
- Objective: Cho LLM khả năng chạy lệnh cmd/PowerShell nhưng chỉ trong whitelist đã định
  nghĩa trong config (ví dụ: mở thư mục, xem thông tin hệ thống).
- Guidance: Tool function validate lệnh/tham số khớp whitelist trước khi `subprocess.run`,
  reject và trả lỗi rõ ràng nếu không khớp. Đăng ký tool này vào `model.act()`.
- Test: Unit test reject lệnh không whitelist; test sanitize tham số; integration test
  chạy 1 lệnh whitelist thật.
- Demo: Lệnh giọng nói "mở thư mục Downloads" → Jarvis thực thi và xác nhận bằng giọng
  nói.

Task 9: App-launch alias tool + media control (PyAutoGUI)
- Objective: Mở ứng dụng đã cài theo alias map (mở nhạc, mở Facebook...) và điều khiển
  media keys (play/pause/volume) qua PyAutoGUI.
- Guidance: Alias map trong config (tên → path/URL/protocol). Tool function resolve alias
  → `os.startfile`/`subprocess`/mở URL trong browser default. Media keys qua
  `pyautogui.press()`.
- Test: Unit test resolve alias hợp lệ/không hợp lệ; test riêng cho hành động media key
  (mock).
- Demo: Lệnh giọng nói "mở Spotify" và "tăng âm lượng" thực thi thành công, có xác nhận
  bằng giọng nói.

Task 10: Web search tool
- Objective: Cho LLM tra cứu thông tin qua DuckDuckGo HTML (hoặc SearxNG nếu bạn tự host),
  tóm tắt kết quả và trả lời bằng giọng nói.
- Guidance: Tool function fetch + parse kết quả tìm kiếm, trả về top N snippet cho LLM
  tóm tắt trong response cuối.
- Test: Unit test parse HTML mock; integration test tìm kiếm thật với 1 câu hỏi đơn giản.
- Demo: Câu hỏi giọng nói "thời tiết Hà Nội hôm nay" → Jarvis tìm kiếm và trả lời tóm tắt
  bằng giọng nói.

Task 11: browser-use tool integration
- Objective: Cho LLM giao tác vụ web automation phức tạp hơn (mở trang, điều hướng, tìm
  nội dung) qua browser-use, dùng cùng LM Studio model làm driver LLM.
- Guidance: Đăng ký browser-use Agent như 1 tool trong `model.act()`, giới hạn phạm vi
  tác vụ đơn giản, set timeout để tránh treo pipeline. Log riêng hoạt động browser-use để
  debug.
- Test: Unit test tool invocation với browser-use agent mock; integration test 1 tác vụ
  browsing đơn giản thật (ví dụ mở YouTube tìm 1 video).
- Demo: Lệnh giọng nói kích hoạt browser-use hoàn thành 1 tác vụ web đơn giản, Jarvis xác
  nhận kết quả bằng giọng nói.

Task 12: Hardening, error handling & setup docs
- Objective: Xử lý lỗi toàn pipeline (LM Studio offline, mic lỗi, STT/TTS timeout, tool
  exception), thêm logging có cấu trúc, tray icon trạng thái lỗi, và README hướng dẫn
  setup từ máy Windows sạch.
- Guidance: Wrap các bước pipeline bằng try/except có thông báo fallback bằng giọng nói
  ("Xin lỗi, tôi gặp lỗi..."), không để crash toàn app khi 1 tool lỗi.
- Test: Fault-injection test (giả lập LM Studio offline, giả lập lỗi mic) assert pipeline
  fallback đúng, không crash.
- Demo: Chạy full demo "Hey Jarvis" → thực hiện tác vụ → trả lời giọng nói, và demo case
  lỗi (tắt LM Studio) cho thấy Jarvis báo lỗi bằng giọng nói thay vì crash.
