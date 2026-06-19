import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'src'))

import unittest
from memory_store import estimate_tokens, UserProfileStore, CompactMemoryManager
from agent_baseline import BaselineAgent
from agent_advanced import AdvancedAgent
from config import load_config


def tmp_state_dir():
    return Path(__file__).resolve().parents[1] / 'tmp_state'


class TestMemoryHelpers(unittest.TestCase):
    def test_estimate_tokens(self):
        assert estimate_tokens('') == 0
        assert estimate_tokens('hello') >= 1
        assert estimate_tokens('a' * 100) >= 20

    def test_user_profile_store(self):
        sd = tmp_state_dir()
        store = UserProfileStore(root_dir=sd)
        user = 'test_user'
        p = store.write_text(user, '# hi')
        assert p.exists()
        txt = store.read_text(user)
        assert '# hi' in txt
        assert store.edit_text(user, '# hi', '# hello')
        assert 'hello' in store.read_text(user)
        assert store.file_size(user) > 0

    def test_compact_memory(self):
        cm = CompactMemoryManager(threshold_tokens=10, keep_messages=2)
        tid = 'thread1'
        for i in range(6):
            cm.append(tid, 'user', 'msg ' + str(i) + ' ' + ('x'*50))
        ctx = cm.context(tid)
        assert 'summary' in ctx
        assert cm.compaction_count(tid) >= 1


class TestBaselineAgent(unittest.TestCase):
    def test_offline_reply_and_accounting(self):
        cfg = load_config()
        ag = BaselineAgent(config=cfg, force_offline=True)
        out = ag.reply('u1', 't1', 'Xin chao')
        assert 'reply' in out
        assert ag.token_usage('t1') >= 0
        assert ag.prompt_token_usage('t1') >= 0

    def test_advanced_offline_persistence(self):
        cfg = load_config()
        ag = AdvancedAgent(config=cfg, force_offline=True)
        user = 'u_adv'
        tid = 't_adv'
        # message that contains a name and profession in Vietnamese
        msg = 'Tên tôi là An. Tôi làm nghề kỹ sư phần mềm.'
        out = ag.reply(user, tid, msg)
        assert 'reply' in out
        # profile file should exist and contain 'An' and 'kỹ sư'
        pf = ag.profile_store.read_text(user)
        assert 'An' in pf
        assert 'kỹ sư' in pf
        # compaction count should be integer
        assert isinstance(ag.compaction_count(tid), int)


if __name__ == '__main__':
    unittest.main()
