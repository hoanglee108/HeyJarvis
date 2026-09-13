"""priority.md P1-1: speak the answer while the model is still writing it.

Two things must hold no matter what the server sends:

* chain-of-thought and tool markup are never spoken, and
* nothing is spoken twice - the fallback to a normal ``say`` only happens when streaming
  produced no audio at all.

Both are asserted here rather than left to manual listening.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from jarvis.streaming import SentenceStream, StreamingSpeaker


class FakeTts:
    """Records what was spoken, in order, and can be told to fail."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.said: list[str] = []
        self.fail_on = fail_on
        self.lock = threading.Lock()

    def speak(self, text: str):  # noqa: ANN202
        with self.lock:
            self.said.append(text)
        if self.fail_on is not None and self.fail_on in text:
            raise RuntimeError("tổng hợp lỗi")
        return type("R", (), {"samples": np.zeros(4), "sample_rate": 22050})()


def feed_all(stream: SentenceStream, chunks: list[str]) -> list[str]:
    spoken: list[str] = []
    for chunk in chunks:
        spoken.extend(stream.feed(chunk))
    spoken.extend(stream.flush())
    return spoken


# -- SentenceStream --------------------------------------------------------------------
def test_complete_sentence_is_emitted_as_soon_as_it_ends() -> None:
    stream = SentenceStream()
    assert stream.feed("Hôm nay trời khá đẹp.") == ["Hôm nay trời khá đẹp."]


def test_partial_sentence_is_held_until_it_finishes() -> None:
    stream = SentenceStream()
    assert stream.feed("Hôm nay trời") == []
    assert stream.feed(" khá đẹp và nắng.") == ["Hôm nay trời khá đẹp và nắng."]


def test_several_sentences_arrive_in_order() -> None:
    stream = SentenceStream()
    spoken = feed_all(stream, ["Trời hôm nay rất đẹp. Nắng nhẹ và dễ chịu. Bạn nên ra ngoài."])
    assert spoken == [
        "Trời hôm nay rất đẹp.",
        "Nắng nhẹ và dễ chịu.",
        "Bạn nên ra ngoài.",
    ]


def test_decimals_do_not_split_a_sentence() -> None:
    """A period with no whitespace after it is not the end of a sentence."""
    stream = SentenceStream()
    spoken = feed_all(stream, ["Nhiệt độ hiện tại là 28.5 độ C và độ ẩm 70.2 phần trăm."])
    assert spoken == ["Nhiệt độ hiện tại là 28.5 độ C và độ ẩm 70.2 phần trăm."]


def test_short_fragments_are_merged_rather_than_spoken_alone() -> None:
    stream = SentenceStream()
    spoken = feed_all(stream, ["Vâng. Tôi đã mở thư mục tải xuống cho bạn rồi."])
    assert spoken == ["Vâng. Tôi đã mở thư mục tải xuống cho bạn rồi."]


def test_question_and_exclamation_end_a_sentence() -> None:
    stream = SentenceStream()
    spoken = feed_all(stream, ["Bạn muốn tôi mở bài nào? Tôi sẵn sàng nghe!"])
    assert spoken == ["Bạn muốn tôi mở bài nào?", "Tôi sẵn sàng nghe!"]


def test_flush_speaks_a_reply_that_never_got_punctuation() -> None:
    stream = SentenceStream()
    assert stream.feed("Đã mở ứng dụng cho bạn") == []
    assert stream.flush() == ["Đã mở ứng dụng cho bạn"]


def test_flush_is_empty_when_nothing_is_pending() -> None:
    stream = SentenceStream()
    stream.feed("Xong rồi nhé bạn ơi.")
    assert stream.flush() == []


# -- reasoning and markup must never be spoken -----------------------------------------
def test_nothing_is_emitted_while_a_think_block_is_open() -> None:
    stream = SentenceStream()
    assert stream.feed("<think>The user wants the time.") == []
    assert stream.feed(" I should call a tool. Let me think more.") == []


def test_closed_think_block_is_dropped_and_the_answer_survives() -> None:
    stream = SentenceStream()
    spoken = feed_all(
        stream,
        ["<think>reasoning in English", " goes here</think>", "Bây giờ là 9 giờ sáng."],
    )
    assert spoken == ["Bây giờ là 9 giờ sáng."]
    assert not any("reasoning" in sentence for sentence in spoken)


def test_tool_markup_leaking_into_content_is_not_spoken() -> None:
    stream = SentenceStream()
    spoken = feed_all(
        stream,
        ["<tool_call>{\"name\": \"get_time\"}</tool_call>", "Đã kiểm tra giúp bạn rồi nhé."],
    )
    assert spoken == ["Đã kiểm tra giúp bạn rồi nhé."]
    assert not any("get_time" in sentence for sentence in spoken)


def test_a_half_typed_tag_is_held_back() -> None:
    """``"<thi"`` could become ``<think>``; emitting on the guess would leak reasoning."""
    stream = SentenceStream()
    assert stream.feed("Xong rồi nhé bạn. <thi") == []


def test_dangling_open_think_is_dropped_on_flush() -> None:
    stream = SentenceStream()
    stream.feed("<think>bị cắt giữa dòng vì hết max_tokens")
    assert stream.flush() == []


# -- StreamingSpeaker ------------------------------------------------------------------
def test_speaker_plays_sentences_in_order() -> None:
    tts = FakeTts()
    speaker = StreamingSpeaker(tts)  # type: ignore[arg-type]
    speaker.start()
    speaker.push("Câu thứ nhất khá dài. ")
    speaker.push("Câu thứ hai cũng vậy. ")
    speaker.push("Và câu cuối cùng ở đây.")

    assert speaker.finish(timeout=5.0)
    assert tts.said == [
        "Câu thứ nhất khá dài.",
        "Câu thứ hai cũng vậy.",
        "Và câu cuối cùng ở đây.",
    ]


def test_speaker_reports_first_audio_once() -> None:
    tts = FakeTts()
    calls: list[int] = []
    speaker = StreamingSpeaker(tts, on_first_audio=lambda: calls.append(1))  # type: ignore[arg-type]
    speaker.start()
    speaker.push("Câu đầu tiên ở đây. Câu thứ hai ở đây nữa.")
    speaker.finish(timeout=5.0)
    assert calls == [1], "chỉ báo một lần khi bắt đầu phát"


def test_speaker_speaks_nothing_when_the_reply_is_only_markup() -> None:
    tts = FakeTts()
    speaker = StreamingSpeaker(tts)  # type: ignore[arg-type]
    speaker.start()
    speaker.push("<think>only reasoning here</think>")
    assert speaker.finish(timeout=5.0) is False
    assert tts.said == []
    assert speaker.spoken_any is False


def test_discard_pending_drops_unspoken_text() -> None:
    """A round that turns into a tool call must not have its preamble spoken."""
    tts = FakeTts()
    speaker = StreamingSpeaker(tts)  # type: ignore[arg-type]
    speaker.start()
    speaker.push("Để tôi kiểm tra giúp bạn")  # no terminal punctuation yet
    speaker.discard_pending()
    speaker.finish(timeout=5.0)
    assert tts.said == []


def test_synthesis_failure_is_reported_without_killing_the_turn() -> None:
    tts = FakeTts(fail_on="thứ hai")
    speaker = StreamingSpeaker(tts)  # type: ignore[arg-type]
    speaker.start()
    speaker.push("Câu thứ nhất khá dài. Câu thứ hai sẽ lỗi ở đây. Câu thứ ba dài dài.")
    ok = speaker.finish(timeout=5.0)

    assert ok is False
    assert speaker.error is not None
    assert speaker.spoken_any is True, "câu đầu đã phát trước khi lỗi"
    assert "Câu thứ ba dài dài." not in tts.said, "phải dừng sau khi lỗi"


def test_finish_is_idempotent() -> None:
    tts = FakeTts()
    speaker = StreamingSpeaker(tts)  # type: ignore[arg-type]
    speaker.start()
    speaker.push("Một câu đủ dài để đọc.")
    assert speaker.finish(timeout=5.0)
    assert speaker.finish(timeout=5.0)
    assert tts.said == ["Một câu đủ dài để đọc."]


def test_ensure_closed_stops_the_worker_thread() -> None:
    tts = FakeTts()
    speaker = StreamingSpeaker(tts)  # type: ignore[arg-type]
    speaker.start()
    worker = speaker._worker  # noqa: SLF001
    speaker.ensure_closed()
    assert worker is not None
    worker.join(timeout=5.0)
    assert not worker.is_alive(), "không được để rò luồng khi lượt nói thất bại"


def test_ensure_closed_after_finish_is_a_no_op() -> None:
    tts = FakeTts()
    speaker = StreamingSpeaker(tts)  # type: ignore[arg-type]
    speaker.start()
    speaker.push("Một câu đủ dài để đọc.")
    speaker.finish(timeout=5.0)
    speaker.ensure_closed()
    assert tts.said == ["Một câu đủ dài để đọc."]


def test_push_after_abort_is_ignored() -> None:
    tts = FakeTts()
    speaker = StreamingSpeaker(tts)  # type: ignore[arg-type]
    speaker.start()
    speaker.abort()
    speaker.push("Câu này không được phát ra.")
    speaker.finish(timeout=5.0)
    assert tts.said == []


def test_text_property_joins_what_was_spoken() -> None:
    tts = FakeTts()
    speaker = StreamingSpeaker(tts)  # type: ignore[arg-type]
    speaker.start()
    speaker.push("Câu một khá là dài. Câu hai cũng khá dài.")
    speaker.finish(timeout=5.0)
    assert speaker.text == "Câu một khá là dài. Câu hai cũng khá dài."


def test_words_are_not_glued_together_at_chunk_boundaries() -> None:
    """Regression: whitespace normalisation used to drop a chunk's trailing space."""
    stream = SentenceStream()
    spoken = feed_all(stream, ["Trời hôm ", "nay khá ", "đẹp và nắng nhẹ."])
    assert spoken == ["Trời hôm nay khá đẹp và nắng nhẹ."]


def test_no_word_is_glued_when_the_stream_is_split_every_character() -> None:
    text = "Bây giờ là chín giờ sáng. Trời hôm nay khá đẹp."
    stream = SentenceStream()
    spoken = feed_all(stream, list(text))
    assert " ".join(spoken) == text
