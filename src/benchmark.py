from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Student TODO: read JSON conversations from disk."""
    import json

    with path.open(encoding="utf8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Student TODO: return 0 / 0.5 / 1 depending on how many expected facts appear."""
    if not expected:
        return 0.0
    found = 0
    a = (answer or "").lower()
    for e in expected:
        if e.lower() in a:
            found += 1
    return float(found) / float(len(expected))


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Student TODO: add a lightweight quality score for offline mode."""
    # simple heuristic: reward recall and brevity
    if not answer:
        return 0.0
    r = recall_points(answer, expected)
    length = len(answer.split())
    # prefer concise answers: penalize very long replies
    brevity = 1.0 if length < 200 else max(0.0, 200.0 / float(length))
    return 0.7 * r + 0.3 * brevity


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Student TODO: evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """

    # track per-conversation metrics
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    compactions_total = 0

    # snapshot memory sizes per user
    users = {c.get("user_id") for c in conversations}
    mem_before = {u: agent.memory_file_size(u) if hasattr(agent, "memory_file_size") else 0 for u in users}

    for conv in conversations:
        conv_id = conv.get("id")
        user_id = conv.get("user_id")
        thread_id = conv_id
        # feed turns
        for t in conv.get("turns", []):
            agent.reply(user_id, thread_id, t)

        # collect tokens for this thread
        if hasattr(agent, "token_usage"):
            total_agent_tokens += int(agent.token_usage(thread_id) or 0)
        if hasattr(agent, "prompt_token_usage"):
            total_prompt_tokens += int(agent.prompt_token_usage(thread_id) or 0)

        # run recall questions in a fresh thread to test cross-session recall
        for i, rq in enumerate(conv.get("recall_questions", [])):
            q = rq.get("question")
            expected = rq.get("expected_contains", [])
            recall_thread = f"{conv_id}-recall-{i}"
            out = agent.reply(user_id, recall_thread, q)
            reply_text = out.get("reply", "") if isinstance(out, dict) else str(out)
            r = recall_points(reply_text, expected)
            qscore = heuristic_quality(reply_text, expected)
            recall_scores.append(r)
            quality_scores.append(qscore)

        # compactions per thread
        if hasattr(agent, "compaction_count"):
            compactions_total += int(agent.compaction_count(thread_id) or 0)

    # memory growth
    mem_after = {u: agent.memory_file_size(u) if hasattr(agent, "memory_file_size") else 0 for u in users}
    memory_growth = sum(max(0, mem_after[u] - mem_before.get(u, 0)) for u in users)

    recall_avg = float(sum(recall_scores) / len(recall_scores)) if recall_scores else 0.0
    quality_avg = float(sum(quality_scores) / len(quality_scores)) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=int(total_agent_tokens),
        prompt_tokens_processed=int(total_prompt_tokens),
        recall_score=recall_avg,
        response_quality=quality_avg,
        memory_growth_bytes=int(memory_growth),
        compactions=int(compactions_total),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Student TODO: print a markdown table or tabulated output."""
    lines = []
    header = ["Agent","Agent tokens only","Prompt tokens processed","Cross-session recall","Response quality","Memory growth (bytes)","Compactions"]
    lines.append(" | ".join(header))
    lines.append(" | ".join(["---"]*len(header)))
    for r in rows:
        lines.append(" | ".join([
            r.agent_name,
            str(r.agent_tokens_only),
            str(r.prompt_tokens_processed),
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            str(r.memory_growth_bytes),
            str(r.compactions),
        ]))
    return "\n".join(lines)


def main() -> None:
    """Student TODO: run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    repo = Path(__file__).resolve().parent.parent
    config = load_config(repo)

    data_standard = load_conversations(repo / 'data' / 'conversations.json')
    data_stress = load_conversations(repo / 'data' / 'advanced_long_context.json')

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    print('Running standard benchmark:')
    rows = []
    rows.append(run_agent_benchmark('baseline', baseline, data_standard, config))
    rows.append(run_agent_benchmark('advanced', advanced, data_standard, config))
    print(format_rows(rows))

    print('\nRunning stress benchmark:')
    rows2 = []
    baseline2 = BaselineAgent(config=config, force_offline=True)
    advanced2 = AdvancedAgent(config=config, force_offline=True)
    rows2.append(run_agent_benchmark('baseline', baseline2, data_stress, config))
    rows2.append(run_agent_benchmark('advanced', advanced2, data_stress, config))
    print(format_rows(rows2))


if __name__ == "__main__":
    main()
