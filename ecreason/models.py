"""Model backends for the cascade evaluator.

Three backends share one interface:  act(prompt, valid_labels) -> (label, conf, raw)

* MockBackend         - no API/GPU. For plumbing tests and as a calibratable oracle.
                        It is given the gold label(s) per step via `set_gold(...)` and
                        returns a correct label with probability `p_level[level]`.
* OpenAICompatBackend - any OpenAI-compatible chat API (needs API key + base_url).  [API]
* HFBackend           - local HuggingFace model via transformers (needs GPU + weights). [GPU]

Only MockBackend runs with zero external dependencies.
"""
from __future__ import annotations

import os
import random

from .prompts import parse_answer


class MockBackend:
    """Oracle-ish simulator: picks a correct label w.p. p(level), else random.

    Useful to (a) smoke-test the harness end to end, and (b) draw an *upper-bound-ish*
    curve as a function of per-level accuracy. It does NOT read the prompt; gold labels
    are injected by the driver via set_gold().
    """

    name = "mock"

    def __init__(self, p_level=None, seed: int = 0):
        # default: easy at L1, harder deeper — mimics the observed LLM degradation shape
        self.p_level = p_level or {1: 0.85, 2: 0.6, 3: 0.45, 4: 0.3}
        self.rng = random.Random(seed)
        self._gold: list[str] = []
        self._level = 1

    def set_gold(self, gold_labels: list[str], level: int):
        self._gold = gold_labels
        self._level = level

    def act(self, prompt: str, valid_labels: list[str]):
        p = self.p_level.get(self._level, 0.3)
        if self._gold and self.rng.random() < p:
            label = self.rng.choice(self._gold)
        else:
            label = self.rng.choice(valid_labels)
        conf = round(self.rng.uniform(0.4, 0.95), 2)
        return label, conf, f"(mock) ANSWER={label} ; CONF={conf}"

    # --- zero-shot (free-form) simulation, for plumbing the baseline ---
    def set_zeroshot_gold(self, ecs: list[str]):
        self._zs_gold = ecs or []

    def generate(self, prompt: str) -> str:
        """Mimic a vanilla LLM doing direct EC prediction: rarely right, often a
        plausible-looking but invalid/garbled EC (drives valid-EC rate < 1)."""
        gold = getattr(self, "_zs_gold", [])
        r = self.rng.random()
        if gold and r < 0.20:                      # correct
            return f"EC={self.rng.choice(gold)}"
        if gold and r < 0.60:                      # right class, hallucinated leaf
            head = gold[0].split(".")[0]
            return f"EC={head}.{self.rng.randint(1, 99)}.{self.rng.randint(1, 99)}.{self.rng.randint(1, 99)}"
        return "I'm not fully certain, but it appears to be an enzyme."  # no parsable EC


class OpenAICompatBackend:
    """OpenAI-compatible chat completions.

    Works with OpenAI, DeepSeek, Qwen-API, Together, or a LOCAL vLLM/TGI server that
    exposes the OpenAI schema. Reads from env (so `run/api/.env` just works):
      OPENAI_API_KEY   - required for remote APIs; any value for local servers
      OPENAI_BASE_URL  - set this to point at a custom/local endpoint (e.g. http://localhost:8000/v1)
    """

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 temperature: float | None = None, max_tokens: int | None = None, max_retries: int = 4):
        self.model = model
        self.name = f"openai:{model}"
        # Reasoning models (GLM / DeepSeek-R / etc.) spend tokens on a hidden chain of
        # thought before emitting the answer in `content`; too small a budget leaves
        # `content` empty (every parse fails -> fake 0 score). Env-overridable so the
        # .env can tune the budget without code edits.
        self.temperature = temperature if temperature is not None else float(os.environ.get("OPENAI_TEMPERATURE", "0.0"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.environ.get("OPENAI_MAX_TOKENS", "3072"))
        self.max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", str(max_retries)))
        # Optional thinking toggle for reasoning models on SGLang/vLLM-style servers.
        # Unset -> use server default (thinking ON). "false" -> fast, answer straight to `content`.
        et = os.environ.get("OPENAI_ENABLE_THINKING")
        self.enable_thinking = None if et is None else (et.strip().lower() in ("1", "true", "yes", "on"))
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install openai") from e
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        key = os.environ.get(api_key_env) or ("EMPTY" if base_url else None)
        if not key:
            raise RuntimeError(
                f"Set {api_key_env} (and optionally OPENAI_BASE_URL) to use the API backend. "
                "See run/api/.env.example.")
        self.client = OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)

    @staticmethod
    def _coerce_text(content) -> str:
        """Normalize a message field to text. Some providers (Gemini / Claude via
        OpenAI-compat gateways) return `content` as a LIST of parts, e.g.
        [{"type": "text", "text": "..."}], instead of a plain string."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            out = []
            for p in content:
                if isinstance(p, str):
                    out.append(p)
                elif isinstance(p, dict):
                    out.append(p.get("text") or p.get("content") or "")
                else:
                    t = getattr(p, "text", None)
                    out.append(t if isinstance(t, str) else "")
            return "".join(out)
        return str(content)

    @staticmethod
    def _message_text(msg) -> str:
        """Prefer the final answer in `content`; fall back to a reasoning model's
        `reasoning_content` when `content` is empty (e.g. thinking truncated)."""
        text = OpenAICompatBackend._coerce_text(getattr(msg, "content", None)).strip()
        if text:
            return text
        rc = getattr(msg, "reasoning_content", None)
        if not rc:
            extra = getattr(msg, "model_extra", None) or {}
            rc = extra.get("reasoning_content")
        return OpenAICompatBackend._coerce_text(rc).strip()

    def _chat(self, prompt: str) -> str:
        import time as _time

        extra = ({"extra_body": {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}}
                 if self.enable_thinking is not None else {})
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **extra,
                )
                return self._message_text(resp.choices[0].message)
            except Exception as e:  # rate limit / transient
                last_err = e
                _time.sleep(min(2 ** attempt, 45))
        raise RuntimeError(f"API call failed after {self.max_retries} retries: {last_err}")

    def act(self, prompt: str, valid_labels: list[str]):
        raw = self._chat(prompt)
        label, conf = parse_answer(raw, valid_labels)
        return label, conf, raw

    def generate(self, prompt: str) -> str:
        """Free-form completion (for the zero-shot direct-prediction baseline)."""
        return self._chat(prompt)


class HFBackend:
    """Local HuggingFace causal LM.  [requires GPU + downloaded weights]"""

    def __init__(self, model_id: str, max_new_tokens: int = 512, temperature: float = 0.0):
        self.name = f"hf:{model_id}"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        import torch  # noqa
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

    def generate(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        inputs = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(inputs, max_new_tokens=self.max_new_tokens,
                                  do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-5))
        return self.tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    def act(self, prompt: str, valid_labels: list[str]):
        raw = self.generate(prompt)
        label, conf = parse_answer(raw, valid_labels)
        return label, conf, raw


def build_backend(spec: str):
    """spec: 'mock' | 'openai:<model>' | 'hf:<model_id>'.

    For 'openai:*' the base_url is taken from $OPENAI_BASE_URL (lets the SAME backend
    drive a remote API or a local vLLM/TGI server).
    """
    if spec == "mock":
        return MockBackend()
    if spec.startswith("openai:"):
        return OpenAICompatBackend(model=spec.split(":", 1)[1], base_url=os.environ.get("OPENAI_BASE_URL"))
    if spec.startswith("hf:"):
        return HFBackend(model_id=spec.split(":", 1)[1])
    raise ValueError(f"unknown backend spec: {spec}")
