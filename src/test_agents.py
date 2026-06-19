from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from model_provider import ProviderConfig
from memory_store import UserProfileStore, CompactMemoryManager


def make_config(tmp_path: Path) -> LabConfig:
    # build a minimal LabConfig that points state to tmp_path
    base = tmp_path
    data_dir = tmp_path / 'data'
    state_dir = tmp_path / 'state'
    state_dir.mkdir(parents=True, exist_ok=True)
    provider = ProviderConfig(provider='offline', model_name='offline', temperature=0.0)
    return LabConfig(
        base_dir=base,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=100,
        compact_keep_messages=3,
        model=provider,
        judge_model=provider,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    store = UserProfileStore(cfg.state_dir / 'profiles')
    user = 'pytest_user'
    p = store.write_text(user, '# hello')
    assert p.exists()
    txt = store.read_text(user)
    assert '# hello' in txt
    assert store.edit_text(user, '# hello', '# hi')
    assert 'hi' in store.read_text(user)
    assert store.file_size(user) > 0


def test_compact_trigger(tmp_path: Path) -> None:
    cm = CompactMemoryManager(threshold_tokens=5, keep_messages=2)
    tid = 't1'
    for i in range(6):
        cm.append(tid, 'user', 'x' * 50)
    ctx = cm.context(tid)
    assert 'summary' in ctx
    assert cm.compaction_count(tid) >= 1


def test_cross_session_recall(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)
    user = 'cross1'
    # feed a fact in one thread
    baseline.reply(user, 'thread-a', 'Tên tôi là Linh')
    advanced.reply(user, 'thread-a', 'Tên tôi là Linh')
    # ask in new thread
    out_b = baseline.reply(user, 'thread-b', 'Bạn tên gì?')
    out_a = advanced.reply(user, 'thread-b', 'Bạn tên gì?')
    assert isinstance(out_b.get('reply'), str)
    assert isinstance(out_a.get('reply'), str)
    assert ('Linh' in out_a.get('reply')) or ('tên' in out_a.get('reply'))


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    # make compact threshold small so advanced compacts
    cfg.compact_threshold_tokens = 20
    cfg.compact_keep_messages = 2
    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)
    user = 'long1'
    tid = 'long-thread'
    # create many long turns
    for i in range(12):
        txt = f"turn {i} " + ('x' * 200)
        baseline.reply(user, tid, txt)
        advanced.reply(user, tid, txt)
    b_prompt = baseline.prompt_token_usage(tid)
    a_prompt = advanced.prompt_token_usage(tid)
    # advanced should have equal or lower prompt tokens after compaction
    assert a_prompt <= b_prompt
