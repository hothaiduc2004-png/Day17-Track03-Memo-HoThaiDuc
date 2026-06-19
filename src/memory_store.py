from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from typing import Optional


def estimate_tokens(text: str) -> int:
    """Student TODO: implement a simple token estimator.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """

    if not text:
        return 0
    s = text.strip()
    if not s:
        return 0
    # rough heuristic: 1 token per ~4 characters
    return max(0, (len(s) + 3) // 4)


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Student TODO:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        # TODO: slugify or sanitize the user id before building the file path.
        safe = re.sub(r"[^a-zA-Z0-9._-]", "_", user_id)
        return self.root_dir / f"User_{safe}.md"

    def read_text(self, user_id: str) -> str:
        # TODO: return file content or an empty default markdown profile.
        p = self.path_for(user_id)
        if not p.exists():
            return f"# User profile: {user_id}\n\n"
        return p.read_text(encoding="utf8")

    def write_text(self, user_id: str, content: str) -> Path:
        # TODO: write markdown to disk and return the file path.
        p = self.path_for(user_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf8")
        return p

    def facts(self, user_id: str) -> dict[str, str]:
        text = self.read_text(user_id)
        facts: dict[str, str] = {}
        for line in text.splitlines():
            if line.startswith("- **") and ":" in line:
                key = line.split("**", 2)[1].strip().lower()
                value = line.split(":", 1)[1].strip()
                if value:
                    facts[key] = value
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> bool:
        p = self.path_for(user_id)
        content = self.read_text(user_id)
        facts = self.facts(user_id)
        normalized_key = key.strip().lower()
        new_line = f"- **{normalized_key}**: {value}"
        if normalized_key in facts:
            old_line = next((line for line in content.splitlines() if line.lower().startswith(f"- **{normalized_key}**")), None)
            if old_line and old_line.strip() != new_line:
                content = content.replace(old_line, new_line, 1)
                self.write_text(user_id, content)
                return True
            return False
        else:
            self.write_text(user_id, content.rstrip() + "\n" + new_line + "\n")
            return True

    def upsert_facts(self, user_id: str, updates: dict[str, str]) -> list[str]:
        changed = []
        for key, value in updates.items():
            if self.upsert_fact(user_id, key, value):
                changed.append(key)
        return changed

    def edit_text(self, user_id: str, old: str, new: str) -> bool:
        content = self.read_text(user_id)
        if old not in content:
            return False
        self.write_text(user_id, content.replace(old, new, 1))
        return True

    def decay_facts(self, user_id: str, max_facts: int = 10) -> None:
        facts = self.facts(user_id)
        if len(facts) <= max_facts:
            return
        lines = [line for line in self.read_text(user_id).splitlines() if line.startswith("- **")]
        keep = lines[-max_facts:]
        header = [line for line in self.read_text(user_id).splitlines() if not line.startswith("- **")]
        new_text = "\n".join(header + keep) + "\n"
        self.write_text(user_id, new_text)

    def file_size(self, user_id: str) -> int:
        # TODO: return the current file size in bytes.
        p = self.path_for(user_id)
        if not p.exists():
            return 0
        return p.stat().st_size


def extract_profile_updates(message: str) -> dict[str, str]:
    """Student TODO: convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    facts: dict[str, str] = {}
    if not message or message.strip().endswith('?'):
        return facts

    patterns = {
        "name": [
            r"tên tôi là\s*([A-Za-zÀ-ỹ\s]+)",
            r"mình tên là\s*([A-Za-zÀ-ỹ\s]+)",
            r"tên mình là\s*([A-Za-zÀ-ỹ\s]+)",
        ],
        "location": [
            r"tôi sống ở\s*([A-Za-z0-9À-ỹ\s,.-]+)",
            r"ở\s*([A-Za-z0-9À-ỹ\s,.-]+)",
            r"hiện tại.*ở\s*([A-Za-z0-9À-ỹ\s,.-]+)",
        ],
        "profession": [
            r"làm nghề\s*([A-Za-zÀ-ỹ\s]+)",
            r"nghề nghiệp.*là\s*([A-Za-zÀ-ỹ\s]+)",
            r"hiện tại.*làm\s*([A-Za-zÀ-ỹ\s]+)",
        ],
        "preference": [
            r"mình muốn bạn trả lời\s*([A-Za-zÀ-ỹ\s,]+)",
            r"style trả lời.*\s*([A-Za-zÀ-ỹ\s,]+)",
            r"phong cách.*\s*([A-Za-zÀ-ỹ\s,]+)",
        ],
        "favorite_drink": [
            r"đồ uống yêu thích.*là\s*([A-Za-zÀ-ỹ\s]+)",
            r"thích\s*([A-Za-zÀ-ỹ\s]+)",
        ],
        "favorite_food": [
            r"món ăn yêu thích.*là\s*([A-Za-zÀ-ỹ\s]+)",
            r"thích ăn\s*([A-Za-zÀ-ỹ\s]+)",
        ],
    }
    for key, regexes in patterns.items():
        for regex in regexes:
            m = re.search(regex, message, re.I)
            if m:
                fact = m.group(1).strip()
                if fact:
                    facts[key] = fact
                    break

    lower = message.lower()
    if "đính chính" in lower or "sửa" in lower or "không còn" in lower:
        if "huế" in lower and "đà nẵng" in lower:
            facts["location"] = "Đà Nẵng"
        if "backend" in lower and "mlops" in lower:
            facts["profession"] = "MLOps engineer"
    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Student TODO: create a compact summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    if not messages:
        return ""
    # Keep the last `max_items` messages and create a short concatenation
    selected = messages[-max_items:]
    parts = []
    for m in selected:
        role = m.get("role", "user")
        content = m.get("content", "")
        snippet = content.replace('\n', ' ')[:200]
        parts.append(f"{role}: {snippet}")
    return "\n".join(parts)


@dataclass
class CompactMemoryManager:
    """Student TODO: implement compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        # TODO:
        # 1. create thread state if missing
        # 2. append the new message
        # 3. trigger compaction if needed
        st = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        msgs: list[dict[str, str]] = st["messages"]
        msgs.append({"role": role, "content": content})

        # estimate total tokens
        total = sum(estimate_tokens(m["content"]) for m in msgs)
        if total > self.threshold_tokens:
            # create a summary from older messages
            keep = self.keep_messages
            summary = summarize_messages(msgs[:-keep], max_items=6)
            # reduce messages to the most recent `keep` messages
            st["summary"] = (st.get("summary", "") + "\n" + summary).strip()
            st["messages"] = msgs[-keep:]
            st["compactions"] = st.get("compactions", 0) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.get(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
