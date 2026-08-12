"""mvp.md Task 7: state machine transitions drive the tray icon."""

from __future__ import annotations

from jarvis.state import State, StateMachine, is_valid_transition, label_for


def test_full_conversation_cycle_is_valid() -> None:
    cycle = [State.IDLE, State.LISTENING, State.THINKING, State.SPEAKING, State.IDLE]
    for current, target in zip(cycle, cycle[1:]):
        assert is_valid_transition(current, target), f"{current} -> {target}"


def test_illegal_transition_is_reported() -> None:
    assert not is_valid_transition(State.IDLE, State.SPEAKING)
    assert not is_valid_transition(State.LISTENING, State.SPEAKING)


def test_error_and_stopped_reachable_from_anywhere() -> None:
    for state in State:
        if state is State.STOPPED:
            continue
        assert is_valid_transition(state, State.ERROR)
        assert is_valid_transition(state, State.STOPPED)


def test_observers_receive_current_state_on_subscribe() -> None:
    machine = StateMachine(State.IDLE)
    seen: list[tuple[State, str]] = []
    machine.subscribe(lambda state, detail: seen.append((state, detail)))
    assert seen == [(State.IDLE, "")]


def test_observers_receive_updates() -> None:
    machine = StateMachine(State.IDLE)
    seen: list[State] = []
    machine.subscribe(lambda state, _detail: seen.append(state))
    machine.set(State.LISTENING)
    machine.set(State.THINKING, "đang gọi LLM")
    assert seen == [State.IDLE, State.LISTENING, State.THINKING]
    assert machine.state is State.THINKING
    assert machine.detail == "đang gọi LLM"


def test_broken_observer_does_not_break_the_pipeline() -> None:
    machine = StateMachine(State.IDLE)

    def boom(_state: State, _detail: str) -> None:
        raise RuntimeError("tray died")

    machine.subscribe(boom)
    machine.set(State.LISTENING)  # must not raise
    assert machine.state is State.LISTENING


def test_every_state_has_a_vietnamese_label() -> None:
    for state in State:
        assert label_for(state)
