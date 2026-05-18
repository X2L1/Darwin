"""Unified Darwin chat brain.

The brain is the single user-facing coordinator. It hides the internal domain
agents behind one conversation surface and decides which system capability to
use for each message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from darwin.core.llm import DarwinLLMEngine


@dataclass
class ChatResponse:
    message: str
    intent: str
    actions: List[Dict[str, Any]] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "intent": self.intent,
            "actions": self.actions,
            "data": self.data,
        }


class UnifiedDarwinBrain:
    """Single chat interface over the model, knowledge base, and improvement loop."""

    def __init__(
        self,
        cfg: Any,
        model: Any,
        tokenizer: Any,
        knowledge_base: Any,
        improvement_loop: Any,
        metrics: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.knowledge_base = knowledge_base
        self.improvement_loop = improvement_loop
        self.metrics = metrics
        self.llm = DarwinLLMEngine(model, tokenizer)

    def chat(
        self,
        message: str,
        max_new_tokens: int = 160,
        temperature: float = 0.8,
        run_improvements: bool = True,
    ) -> ChatResponse:
        intent = _classify_intent(message)

        if intent == "status":
            return self._status_response()
        if intent == "help":
            return self._help_response()
        if intent == "greeting":
            return self._greeting_response()
        if intent == "identity":
            return self._identity_response()
        if intent == "training":
            return self._training_response()
        if intent == "source":
            return self._source_response()
        if intent == "improve":
            if not run_improvements:
                return ChatResponse(
                    message="I understood this as a self-improvement request, but cycle execution is disabled for this call.",
                    intent=intent,
                    actions=[],
                )
            return self._improve_response(message)
        if intent == "knowledge":
            return self._knowledge_response(message)
        return self._chat_response(message, max_new_tokens=max_new_tokens, temperature=temperature)

    def _help_response(self) -> ChatResponse:
        message = (
            "Here is the simple version: talk to me in this box, and I route the work internally.\n\n"
            "Good things to try:\n"
            "- status\n"
            "- show me the model source\n"
            "- improve yourself\n"
            "- search knowledge <topic>\n"
            "- add a file path in the References panel, then ask about it\n\n"
            "The raw transformer is custom and local, but it still needs better training before it can "
            "answer open-ended questions like a polished chatbot. Until then, I use grounded system "
            "responses so the app stays understandable instead of spitting out nonsense."
        )
        return ChatResponse(
            message=message,
            intent="help",
            actions=[{"type": "explain_usage"}],
            data={"suggested_prompts": ["status", "show me the model source", "improve yourself"]},
        )

    def _greeting_response(self) -> ChatResponse:
        return ChatResponse(
            message=(
                "Hey, I am Darwin. I am running locally through the custom transformer wrapper. "
                "You can ask for status, inspect my model source, add reference files, or tell me to "
                "run an improvement cycle."
            ),
            intent="greeting",
            actions=[{"type": "greet"}],
            data={"model": self.llm.describe()},
        )

    def _identity_response(self) -> ChatResponse:
        model_info = self.llm.describe()
        return ChatResponse(
            message=(
                "I am Darwin: a local self-improving AI project with one user-facing chat surface. "
                "Behind this chat I can call internal agents, search your local knowledge base, inspect "
                "my source files, and run guarded improvement cycles. "
                f"The active model provider is {model_info['provider']}, not Ollama."
            ),
            intent="identity",
            actions=[{"type": "describe_self"}],
            data={"model": model_info},
        )

    def _training_response(self) -> ChatResponse:
        return ChatResponse(
            message=(
                "The reason plain chat can feel repetitive is that the custom transformer is present, "
                "but its weights are not yet trained into a capable assistant. The app now falls back to "
                "grounded local responses instead of showing junk text.\n\n"
                "To improve the actual language model, add reference material in the References panel, "
                "run improvement cycles, and train/fine-tune with `python -m darwin.cli train --data-file "
                "<text-file> --small --data-dir data` from the project folder."
            ),
            intent="training",
            actions=[{"type": "explain_model_training_state"}],
            data={"model": self.llm.describe()},
        )

    def _status_response(self) -> ChatResponse:
        metrics_summary = self.metrics.summary() if self.metrics is not None else {}
        model_info = self.llm.describe()
        message = (
            "Darwin is online as one unified assistant. "
            f"The custom model has {model_info['parameters']:,} trainable parameters, "
            f"the knowledge base has {self.knowledge_base.count()} entries, and the model provider is "
            f"{model_info['provider']}."
        )
        return ChatResponse(
            message=message,
            intent="status",
            actions=[{"type": "read_status"}],
            data={
                "model": model_info,
                "kb_entries": self.knowledge_base.count(),
                "metrics": metrics_summary,
            },
        )

    def _source_response(self) -> ChatResponse:
        model_info = self.llm.describe()
        message = (
            "The language model source is available for Darwin's internal code review and patching. "
            "The main implementation lives in darwin/core/model.py, with tokenizer and training code beside it."
        )
        return ChatResponse(
            message=message,
            intent="source",
            actions=[{"type": "inspect_model_source"}],
            data={"model": model_info},
        )

    def _improve_response(self, message: str) -> ChatResponse:
        summary = self.improvement_loop.run_once(
            context_overrides={
                "code": {"user_goal": message},
                "prompting": {"user_goal": message},
                "research": {"user_goal": message},
            }
        )
        reply = (
            "I ran one internal self-improvement cycle. "
            f"It produced {summary['n_proposals_total']} raw proposals, fused them to {summary['n_fused']}, "
            f"merged {summary['n_merged']} safe change(s), and queued "
            f"{summary['n_human_review_queued']} item(s) for review."
        )
        return ChatResponse(
            message=reply,
            intent="improve",
            actions=[{"type": "run_self_improvement_cycle", "summary": summary}],
            data={"cycle": summary},
        )

    def _knowledge_response(self, message: str) -> ChatResponse:
        query = _strip_intent_words(message)
        results = self.knowledge_base.search(query, top_k=5) if query else []
        if results:
            titles = ", ".join(r["title"] for r in results[:3])
            reply = f"I searched the knowledge base and found: {titles}."
        else:
            reply = "I searched the knowledge base and did not find a matching entry yet."
        return ChatResponse(
            message=reply,
            intent="knowledge",
            actions=[{"type": "search_knowledge", "query": query}],
            data={"results": results},
        )

    def _chat_response(self, message: str, max_new_tokens: int, temperature: float) -> ChatResponse:
        context = ""
        if self.knowledge_base.count() > 0:
            context = self.improvement_loop.orchestrator.retriever.get_context(message)
        prompt = _build_prompt(message, context)
        generated = self.llm.complete(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        response_text = generated.text
        grounded = None
        if generated.used_fallback:
            grounded = _grounded_chat_reply(
                message=message,
                context=context,
                model_info=self.llm.describe(),
                kb_count=self.knowledge_base.count(),
            )
            response_text = grounded.text

        actions = [{"type": "generate_with_custom_llm", "used_fallback": generated.used_fallback}]
        if grounded is not None:
            actions.append({"type": "grounded_system_reply", "reason": grounded.reason})
        return ChatResponse(
            message=response_text,
            intent="chat",
            actions=actions,
            data={
                "prompt_tokens": generated.prompt_tokens,
                "completion_tokens": generated.completion_tokens,
                "model": self.llm.describe(),
                "raw_model_text": generated.text if generated.used_fallback else None,
            },
        )


def _classify_intent(message: str) -> str:
    text = message.lower().strip()
    if text in {"hi", "hello", "hey", "yo"} or text.startswith(("hi ", "hello ", "hey ")):
        return "greeting"
    if any(phrase in text for phrase in ("what can you do", "how do i use", "help", "commands")):
        return "help"
    if any(phrase in text for phrase in ("who are you", "what are you", "what is darwin")):
        return "identity"
    if any(phrase in text for phrase in ("same response", "repeating", "fallback", "train", "training", "weights")):
        return "training"
    if any(word in text for word in ("status", "health", "online", "working")):
        return "status"
    if "source" in text or "model.py" in text or "architecture" in text:
        return "source"
    if any(word in text for word in ("improve", "evolve", "self-improvement", "self improve", "fix bug", "fix bugs", "upgrade")):
        return "improve"
    if "knowledge" in text or text.startswith("search "):
        return "knowledge"
    return "chat"


def _strip_intent_words(message: str) -> str:
    cleaned = message
    for phrase in ("search", "knowledge base", "knowledge", "for"):
        cleaned = cleaned.replace(phrase, " ")
    return " ".join(cleaned.split())


def _build_prompt(message: str, context: str) -> str:
    parts = [
        "You are Darwin, one unified self-improving AI assistant.",
        "Answer the user directly. Do not ask the user to choose an internal agent category.",
    ]
    if context:
        parts.append(context)
    parts.append(f"User: {message}")
    parts.append("Darwin:")
    return "\n\n".join(parts)


@dataclass
class GroundedReply:
    text: str
    reason: str


def _grounded_chat_reply(
    message: str,
    context: str,
    model_info: Dict[str, Any],
    kb_count: int,
) -> GroundedReply:
    text = message.lower()

    if context:
        return GroundedReply(
            text=(
                "I found relevant local reference context, but the raw transformer is not confident "
                "enough to summarize it cleanly yet. Try asking `search knowledge "
                f"{message.strip()}` to see the closest stored sources, or add more focused reference "
                "files in the References panel."
            ),
            reason="knowledge_context_present",
        )

    if "?" in message or text.startswith(("why", "how", "what", "can ", "do ")):
        return GroundedReply(
            text=(
                "I can route and explain Darwin-specific things right now, but the open-ended language "
                "model still needs training before it can reliably answer arbitrary questions. "
                "Try asking about my status, source, improvement cycle, or knowledge base. "
                f"Current local model: {model_info['parameters']:,} parameters; references: {kb_count}."
            ),
            reason="untrained_open_ended_question",
        )

    return GroundedReply(
        text=(
            "I heard you. The custom transformer did not produce a reliable answer for that message yet, "
            "so I am using the safer local control layer. You can ask me for `status`, `help`, "
            "`show me the model source`, `improve yourself`, or `search knowledge <topic>`."
        ),
        reason="untrained_general_chat",
    )
