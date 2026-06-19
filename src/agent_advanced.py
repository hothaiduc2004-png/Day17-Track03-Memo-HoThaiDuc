from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Student TODO: implement Agent B / Advanced Agent.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None
        self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: route between offline mode and live mode."""
        if self.langchain_agent and not self.force_offline:
            try:
                prompt = self._build_live_prompt(user_id, thread_id, message)
                reply_text = self.langchain_agent.predict(prompt)
            except Exception:
                reply_text = self.langchain_agent.predict(message)
            result = {"reply": reply_text}
        else:
            result = self._reply_offline(user_id, thread_id, message)

        return {
            "thread_id": thread_id,
            "reply": result.get("reply"),
            "token_usage": self.token_usage(thread_id),
            "prompt_tokens_processed": self.prompt_token_usage(thread_id),
            "memory_file": str(self.profile_store.path_for(user_id)),
        }

    def token_usage(self, thread_id: str) -> int:
        return int(self.thread_tokens.get(thread_id, 0))

    def prompt_token_usage(self, thread_id: str) -> int:
        return int(self.thread_prompt_tokens.get(thread_id, 0))

    def memory_file_size(self, user_id: str) -> int:
        return int(self.profile_store.file_size(user_id))

    def compaction_count(self, thread_id: str) -> int:
        return int(self.compact_memory.compaction_count(thread_id))

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Student TODO: implement the deterministic advanced path.

        Pseudocode:
        1. Extract stable profile facts from the incoming message.
        2. Persist those facts into `User.md`.
        3. Append the message into compact memory.
        4. Estimate prompt-context load from `User.md` + summary + recent messages.
        5. Generate a response that can answer long-term recall questions.
        6. Append the assistant reply and update token counters.
        """

        # 1. extract profile facts
        facts = extract_profile_updates(message)
        if facts:
            self.profile_store.upsert_facts(user_id, facts)
            self.profile_store.decay_facts(user_id, max_facts=10)

        # 2. append message into compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 3. estimate prompt-context tokens
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        # 4. produce a deterministic response using persisted memory
        reply_text = self._offline_response(user_id, thread_id, message)

        # 5. append assistant reply into compact memory
        self.compact_memory.append(thread_id, "assistant", reply_text)

        # 6. update token counters
        reply_tokens = estimate_tokens(reply_text)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + reply_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        return {"reply": reply_text, "prompt_tokens": prompt_tokens, "reply_tokens": reply_tokens}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Student TODO: estimate the context carried into one turn.

        Hint:
        - Include `User.md`
        - Include compact summary text
        - Include recent kept messages
        """

        total = 0
        # include User.md
        profile_text = self.profile_store.read_text(user_id)
        total += estimate_tokens(profile_text)

        # include compact summary and recent messages
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "")
        total += estimate_tokens(summary)
        for m in ctx.get("messages", []):
            total += estimate_tokens(m.get("content", ""))

        return int(total)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Student TODO: return a deterministic answer using persisted memory.

        Make sure the advanced agent can answer questions like:
        - "Mình tên gì?"
        - "Hiện tại mình làm nghề gì?"
        - "Nhắc lại style trả lời mình thích"
        - questions in the long stress dataset
        """

        # simple deterministic responders reading profile
        profile = self.profile_store.read_text(user_id)
        # check for name queries
        if any(tok in message.lower() for tok in ["tên", "mình tên", "tên tôi"]):
            # try to find a name line in profile
            m = None
            for line in profile.splitlines():
                if "**name**" in line.lower() or "**tên**" in line.lower():
                    m = line.split(":", 1)[-1].strip()
                    break
            if m:
                return f"Bạn đã nói tên là {m}"
            return "Mình chưa biết tên bạn — bạn muốn giới thiệu không?"

        # check for profession
        if any(tok in message.lower() for tok in ["nghề", "làm nghề", "công việc"]):
            m = None
            for line in profile.splitlines():
                if "**profession**" in line.lower() or "**nghề**" in line.lower():
                    m = line.split(":", 1)[-1].strip()
                    break
            if m:
                return f"Bạn hiện đang làm: {m}"
            return "Mình chưa biết nghề của bạn. Bạn muốn chia sẻ không?"

        # check for style preference
        if any(tok in message.lower() for tok in ["style", "phong cách", "cách trả lời", "tone"]):
            m = None
            for line in profile.splitlines():
                if "**preference**" in line.lower() or "**style**" in line.lower():
                    m = line.split(":", 1)[-1].strip()
                    break
            if m:
                return f"Bạn muốn mình trả lời theo phong cách: {m}"
            return "Mình chưa có preference ghi lại. Bạn muốn mình trả lời thế nào?"

        # default echo with brief context info
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "").strip()
        add = f"\n(Tóm tắt trước đó: {summary[:200]})" if summary else ""
        return f"Tôi nhớ: {profile.splitlines()[0] if profile else ''} {add}\nTrả lời: {message[:200]}"

    def _maybe_build_langchain_agent(self):
        self.langchain_agent = build_chat_model(self.config.model)
        return self.langchain_agent

    def _build_live_prompt(self, user_id: str, thread_id: str, message: str) -> str:
        profile_text = self.profile_store.read_text(user_id)
        ctx = self.compact_memory.context(thread_id)
        summary = ctx.get("summary", "")
        recent = "\n".join(
            f"{m['role']}: {m['content']}" for m in ctx.get("messages", [])
        )
        return (
            f"User profile:\n{profile_text}\n"
            f"Thread summary:\n{summary}\n"
            f"Recent messages:\n{recent}\n"
            f"New user message:\n{message}"
        )
