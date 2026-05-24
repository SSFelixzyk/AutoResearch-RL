# tests/test_history.py
from src.action_space import sample_random_spec
from src.history import HistoryBuffer

def test_history_buffer_sliding_window():
    buf = HistoryBuffer(max_k=3)
    for i in range(5):
        buf.add(sample_random_spec(seed=i), acc=0.5 + i * 0.05)
    assert len(buf.recent) == 3  # sliding window of 3

def test_history_buffer_tracks_best():
    buf = HistoryBuffer(max_k=10)
    buf.add(sample_random_spec(seed=0), acc=0.70)
    buf.add(sample_random_spec(seed=1), acc=0.85)
    buf.add(sample_random_spec(seed=2), acc=0.75)
    assert abs(buf.best_acc - 0.85) < 1e-6

def test_build_prompt_no_history():
    buf = HistoryBuffer(max_k=10)
    program_md = "# Research Agenda\nPropose a config."
    prompt = buf.build_prompt(program_md, use_history=False)
    assert "Research Agenda" in prompt
    assert "History" not in prompt

def test_build_prompt_with_history():
    buf = HistoryBuffer(max_k=10)
    buf.add(sample_random_spec(seed=0), acc=0.72)
    program_md = "# Research Agenda\nPropose a config."
    prompt = buf.build_prompt(program_md, use_history=True)
    assert "0.720" in prompt
    assert "Best so far" in prompt
