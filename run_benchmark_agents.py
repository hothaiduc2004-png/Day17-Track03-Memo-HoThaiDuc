"""Benchmark harness: run BaselineAgent and AdvancedAgent on the provided conversations.json dataset.

Outputs: results.jsonl and results.csv in the repo root.
"""
from __future__ import annotations

import sys
from pathlib import Path
import json
import csv

SRC = Path(__file__).resolve().parent / 'src'
sys.path.append(str(SRC))

from config import load_config
from agent_baseline import BaselineAgent
from agent_advanced import AdvancedAgent


def load_conversations(path: Path):
    with path.open(encoding='utf8') as f:
        return json.load(f)


def run_agent_on_dataset(agent, conversations, agent_name: str, results: list):
    for conv in conversations:
        conv_id = conv.get('id')
        user_id = conv.get('user_id')
        turns = conv.get('turns', [])
        # use thread per conversation
        thread_id = conv_id
        # send each turn
        for t in turns:
            agent.reply(user_id, thread_id, t)
        # after conversation, run recall questions
        for rq in conv.get('recall_questions', []):
            q = rq.get('question')
            expected = rq.get('expected_contains', [])
            out = agent.reply(user_id, thread_id, q)
            reply_text = out.get('reply', '') if isinstance(out, dict) else str(out)
            # check recall: all expected substrings appear in reply_text (case-insensitive)
            ok = True
            for e in expected:
                if e.lower() not in reply_text.lower():
                    ok = False
                    break
            rec = {
                'agent': agent_name,
                'conv_id': conv_id,
                'user_id': user_id,
                'question': q,
                'expected': expected,
                'reply': reply_text,
                'recall_pass': ok,
                'token_usage': agent.token_usage(thread_id) if hasattr(agent, 'token_usage') else None,
                'prompt_tokens': agent.prompt_token_usage(thread_id) if hasattr(agent, 'prompt_token_usage') else None,
                'memory_file_bytes': agent.memory_file_size(user_id) if hasattr(agent, 'memory_file_size') else None,
                'compactions': agent.compaction_count(thread_id) if hasattr(agent, 'compaction_count') else None,
            }
            results.append(rec)


def write_results(out_path: Path, results: list):
    jsonl = out_path.with_suffix('.jsonl')
    with jsonl.open('w', encoding='utf8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    csvf = out_path.with_suffix('.csv')
    keys = ['agent','conv_id','user_id','question','expected','reply','recall_pass','token_usage','prompt_tokens','memory_file_bytes','compactions']
    with csvf.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            row = {k: json.dumps(r[k], ensure_ascii=False) if isinstance(r.get(k), (list, dict)) else r.get(k) for k in keys}
            writer.writerow(row)


if __name__ == '__main__':
    repo = Path(__file__).resolve().parent
    data_file = repo / 'data' / 'conversations.json'
    if not data_file.exists():
        print('Dataset not found:', data_file)
        raise SystemExit(1)

    conversations = load_conversations(data_file)
    cfg = load_config()

    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    results = []
    print('Running BaselineAgent...')
    run_agent_on_dataset(baseline, conversations, 'baseline', results)
    print('Running AdvancedAgent...')
    run_agent_on_dataset(advanced, conversations, 'advanced', results)

    out_path = repo / 'benchmark_results'
    write_results(out_path, results)
    print('Wrote results to', out_path.with_suffix('.jsonl'), out_path.with_suffix('.csv'))
