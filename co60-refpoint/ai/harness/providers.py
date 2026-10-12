"""Minimal provider clients over HTTPS (stdlib only), each with the same conversation interface.

    conv = make_provider(slot).start(system, user, tools)   # tools: [] for A0, [PYTHON_TOOL] for A1/A2
    turn = conv.step()            # one model call; returns Turn(text, tool_calls, usage, stop, raw_request, raw_response)
    conv.add_tool_results([(call_id, name, output_text), ...])
    conv.add_user(text)

API keys come from the environment (names in ai/models.yaml), never from files in the repository. Requests and
responses are returned verbatim (minus credentials) so the runner can store them in ai/raw/.
"""
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

PYTHON_TOOL = {
    "name": "python",
    "description": ("Run Python 3 code. Returns standard output, standard error and the exit status. numpy and "
                    "scipy are available. No internet. Files in the working directory persist between calls."),
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source"}},
                   "required": ["code"]},
}


@dataclass
class Turn:
    text: str
    tool_calls: list            # [(id, name, args_dict)]
    usage: dict                 # input_tokens, output_tokens, reasoning_tokens (when reported)
    stop: str
    raw_request: dict
    raw_response: dict
    model_reported: str = ""
    wall_s: float = 0.0
    http_attempts: int = 1


def _ssl_context():
    ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    return ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()


def post_json(url, payload, headers, timeout=900, retries=5):
    """POST with retries on 429/5xx and connection errors (exponential backoff). Returns (json, attempts)."""
    data = json.dumps(payload).encode()
    delay = 4.0
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
                return json.loads(r.read()), attempt
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:2000]
            if e.code in (408, 409, 429, 500, 502, 503, 504, 529) and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"connection to {url} failed: {e}") from None


RETRY_CODES = (408, 409, 429, 500, 502, 503, 504, 529)


def in_band_error(resp):
    """An error that OpenRouter returns inside an HTTP 200 body (for example an upstream 429 or idle timeout):
    a top-level "error", or a choice with finish_reason "error". None when the response is a real reply."""
    if resp.get("error"):
        return resp["error"] if isinstance(resp["error"], dict) else {"message": str(resp["error"])}
    ch = (resp.get("choices") or [{}])[0]
    if ch.get("finish_reason") == "error" or ch.get("error"):
        return ch.get("error") or {"message": "finish_reason error", "code": None}
    return None


class Conversation:
    def __init__(self, slot, system, user, tools):
        self.slot, self.system, self.tools = slot, system, tools
        self.model = slot["model_id"]
        self.settings = dict(slot.get("settings") or {})   # copied: init() may pop harness-only keys
        self.key = os.environ.get(slot["api_key_env"], "")
        if not self.key:
            raise RuntimeError(f"environment variable {slot['api_key_env']} is not set")
        self.messages = []
        self.init(user)

    def step(self, retries=6):
        url, headers, payload = self.request()
        t0 = time.time()
        attempts, delay = 0, 10.0
        for k in range(1, retries + 1):
            resp, n = post_json(url, payload, headers)
            attempts += n
            err = in_band_error(resp)
            if err is None:
                break
            if err.get("code") in RETRY_CODES and k < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"in-band error after {attempts} attempts: {json.dumps(err)[:500]}")
        turn = self.parse(resp)
        turn.raw_request, turn.raw_response = payload, resp
        turn.wall_s, turn.http_attempts = round(time.time() - t0, 2), attempts
        self.append_assistant(resp)
        return turn


class AnthropicConversation(Conversation):
    def init(self, user):
        self.messages.append({"role": "user", "content": user})

    def request(self):
        p = {"model": self.model, "max_tokens": self.settings.get("max_tokens", 32000), "system": self.system,
             "messages": self.messages, **{k: v for k, v in self.settings.items() if k != "max_tokens"}}
        if self.tools:
            p["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                          for t in self.tools]
        base = self.slot.get("base_url", "https://api.anthropic.com")
        return f"{base}/v1/messages", {"x-api-key": self.key, "anthropic-version": "2023-06-01"}, p

    def parse(self, r):
        text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
        calls = [(b["id"], b["name"], b.get("input", {})) for b in r.get("content", []) if b.get("type") == "tool_use"]
        u = r.get("usage", {})
        return Turn(text, calls, {"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
                                  "cache_read_tokens": u.get("cache_read_input_tokens", 0)},
                    r.get("stop_reason", ""), {}, {}, r.get("model", ""))

    def append_assistant(self, r):
        self.messages.append({"role": "assistant", "content": r.get("content", [])})

    def add_tool_results(self, results):
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": cid, "content": out} for cid, _, out in results]})

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})


class OpenAIChatConversation(Conversation):
    """OpenAI Chat Completions, and any OpenAI-compatible endpoint (open-weight model hosts)."""

    def init(self, user):
        role = self.settings.pop("system_role", "system") if isinstance(self.settings, dict) else "system"
        self.cache = self.settings.pop("prompt_cache", False)
        self.messages += [{"role": role, "content": self.system}, {"role": "user", "content": user}]

    def request(self):
        msgs = self.messages
        if self.cache:
            # Anthropic prompt caching through OpenRouter: breakpoints on the system prompt and the newest message,
            # so each turn reads the conversation so far from the cache. What the model sees is unchanged.
            msgs = [dict(m) for m in self.messages]
            for i in (0, len(msgs) - 1):
                if isinstance(msgs[i].get("content"), str) and msgs[i]["content"]:
                    msgs[i]["content"] = [{"type": "text", "text": msgs[i]["content"],
                                           "cache_control": {"type": "ephemeral"}}]
        p = {"model": self.model, "messages": msgs, **self.settings}
        if self.tools:
            p["tools"] = [{"type": "function", "function": t} for t in self.tools]
        base = self.slot.get("base_url", "https://api.openai.com/v1")
        return f"{base}/chat/completions", {"authorization": f"Bearer {self.key}"}, p

    def parse(self, r):
        ch = (r.get("choices") or [{}])[0]
        m = ch.get("message", {})
        calls = []
        for tc in m.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed_arguments": tc["function"].get("arguments")}
            calls.append((tc["id"], tc["function"]["name"], args))
        u = r.get("usage", {}) or {}
        det = u.get("completion_tokens_details") or {}
        return Turn(m.get("content") or "", calls,
                    {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
                     "reasoning_tokens": det.get("reasoning_tokens", 0)},
                    ch.get("finish_reason", ""), {}, {}, r.get("model", ""))

    def append_assistant(self, r):
        m = dict((r.get("choices") or [{}])[0].get("message", {}))
        m.setdefault("role", "assistant")
        self.messages.append({k: v for k, v in m.items() if v is not None})

    def add_tool_results(self, results):
        for cid, _, out in results:
            self.messages.append({"role": "tool", "tool_call_id": cid, "content": out})

    def add_user(self, text):
        self.messages.append({"role": "user", "content": text})


class GeminiConversation(Conversation):
    def init(self, user):
        self.messages.append({"role": "user", "parts": [{"text": user}]})

    def request(self):
        p = {"systemInstruction": {"parts": [{"text": self.system}]}, "contents": self.messages}
        if self.settings:
            p["generationConfig"] = self.settings
        if self.tools:
            p["tools"] = [{"functionDeclarations": self.tools}]
        base = self.slot.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
        return f"{base}/models/{self.model}:generateContent", {"x-goog-api-key": self.key}, p

    def parse(self, r):
        cand = (r.get("candidates") or [{}])[0]
        parts = cand.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        calls = [(f"call_{i}", p["functionCall"]["name"], p["functionCall"].get("args", {}))
                 for i, p in enumerate(parts) if "functionCall" in p]
        u = r.get("usageMetadata", {})
        return Turn(text, calls, {"input_tokens": u.get("promptTokenCount", 0),
                                  "output_tokens": u.get("candidatesTokenCount", 0),
                                  "reasoning_tokens": u.get("thoughtsTokenCount", 0)},
                    cand.get("finishReason", ""), {}, {}, r.get("modelVersion", ""))

    def append_assistant(self, r):
        cand = (r.get("candidates") or [{}])[0]
        self.messages.append(cand.get("content", {"role": "model", "parts": []}))

    def add_tool_results(self, results):
        self.messages.append({"role": "user", "parts": [
            {"functionResponse": {"name": name, "response": {"result": out}}} for _, name, out in results]})

    def add_user(self, text):
        self.messages.append({"role": "user", "parts": [{"text": text}]})


KINDS = {"anthropic": AnthropicConversation, "openai_chat": OpenAIChatConversation,
         "openai_compatible": OpenAIChatConversation, "gemini": GeminiConversation}


def start(slot, system, user, tools):
    if slot["api"] == "mock":
        from ai.harness.mock import MockConversation
        return MockConversation(slot, system, user, tools)
    return KINDS[slot["api"]](slot, system, user, tools)
