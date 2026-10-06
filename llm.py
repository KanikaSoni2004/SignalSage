"""
llm.py
------
Tiny, dependency-light LLM backends. Each one exposes:  .name  and  .chat(messages) -> str
(`messages` = [{"role": "system"|"user"|"assistant", "content": "..."}], the return value
is the model's raw text, which the agent expects to be ONE JSON object).

Backends
  OllamaBackend        local model via Ollama (no account, no API key). Default: IBM Granite.
  OpenAICompatBackend  any OpenAI-compatible endpoint (Groq, OpenAI, OpenRouter, etc.).
  MockBackend          scripted fake "LLM" so you can test the whole pipeline with no model.
"""
import json
import os
import requests

try:
    import streamlit as st
except ImportError:
    st = None


def _get_secret_or_env(key, default=""):
    """Safely retrieves keys from Streamlit secrets or OS environment variables."""
    if st is not None:
        try:
            if key in st.secrets:
                return str(st.secrets[key])
        except Exception:
            pass
    return os.getenv(key, default)


class OllamaBackend:
    def __init__(self, model="granite4", host=None, num_ctx=4096, timeout=900):
        host = host or _get_secret_or_env("OLLAMA_HOST", "http://localhost:11434")
        if not host.startswith("http"):
            host = "http://" + host
        self.host, self.model, self.num_ctx, self.timeout = host, model, num_ctx, timeout
        self.name = f"ollama-{model}".replace(":", "-").replace("/", "-")

    def chat(self, messages):
        r = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": "json",                     # forces valid JSON output
                "keep_alive": "30m",                  # keep the model loaded between calls
                "options": {
                    "temperature": 0.0,
                    "seed": 0,
                    "num_ctx": self.num_ctx,
                    "num_predict": 400,               # cap reply length (reports are short)
                },
            },
            timeout=self.timeout,
        )
        if r.status_code != 200:
            hint = (
                f"\nModel '{self.model}' is not installed. Run:  ollama pull {self.model}"
                "\n(see installed names with:  ollama list)"
            ) if r.status_code == 404 else ""
            raise RuntimeError(f"Ollama returned HTTP {r.status_code}: {r.text[:300]}{hint}")
        return r.json()["message"]["content"]


class OpenAICompatBackend:
    def __init__(self, model="llama-3.3-70b-versatile", base_url=None, api_key=None, json_mode=False, timeout=300):
        # Resolve Base URL
        groq_key = _get_secret_or_env("GROQ_API_KEY")
        default_base_url = "https://api.groq.com/openai/v1" if groq_key else "https://api.openai.com/v1"
        
        raw_base = base_url or _get_secret_or_env("OPENAI_BASE_URL", default_base_url)
        # Ensure base_url strictly ends at /v1 (no trailing slashes or /chat/completions)
        self.base_url = raw_base.rstrip("/").removesuffix("/chat/completions")
        
        raw_key = api_key or groq_key or _get_secret_or_env("LLM_API_KEY", "")
        self.api_key = raw_key.strip() if raw_key else ""
        self.model, self.json_mode, self.timeout = model, json_mode, timeout
        self.name = f"openai-{model}".replace(":", "-").replace("/", "-")

    def chat(self, messages):
        if not self.api_key:
            raise ValueError(
                "No API Key found. Please add GROQ_API_KEY or LLM_API_KEY to Streamlit Secrets."
            )
        
        body = {"model": self.model, "messages": messages, "temperature": 0}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        r = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class MockBackend:
    """
    Pretends to be an LLM, only to test the pipeline.
      mode="good"   : calls the ML tool, then reports its numbers faithfully.
      mode="sloppy" : first final report contains a WRONG number; it fixes it only
                      after the verifier complains (tests the verifier loop).
      mode="disobedient": first final report has the WRONG verdict; fixes it when told.
      mode="stubborn"   : always gives the WRONG verdict (tests the policy override).
      mode="hallucinate": first explanation asserts unsupported conditions (flat readings, clipping);
                          neutral once the verifier objects (tests the explanation-support check).
    """
    def __init__(self, mode="good"):
        self.mode = mode
        self.name = f"mock-{mode}"

    @staticmethod
    def _last_tool_result(messages, tool):
        for m in reversed(messages):
            if m["role"] != "user":
                continue
            c = m["content"]
            if c.startswith("TOOL_RESULTS "):                     # several tools in one turn
                d = json.loads(c[len("TOOL_RESULTS "):])
                if tool in d:
                    return d[tool]
            elif c.startswith(f"TOOL_RESULT {tool} "):
                return json.loads(c.split(" ", 2)[2])
        return None

    def chat(self, messages):
        sys_prompt = messages[0]["content"]
        if "NO_TOOLS" in sys_prompt:                                  # LLM-only configuration
            return json.dumps({
                "action": "final",
                "report": {
                    "verdict": "unsure",
                    "fault_type": None,
                    "confidence": 0.3,
                    "evidence": [],
                    "explanation": "mock model cannot judge raw numbers.",
                    "next_step": "inspect manually",
                },
            })
        res = self._last_tool_result(messages, "ml_fault_probability")
        if res is None:
            return json.dumps({
                "action": "call_tool",
                "tools": ["ml_fault_probability", "compare_to_history", "check_stuck", "check_spikes"],
            })
        p = res["fault_probability"]
        corrected = any(m["role"] == "user" and m["content"].startswith("VERIFIER") for m in messages)
        shown = p if (self.mode != "sloppy" or corrected) else round(min(1.0, p + 0.2), 4)
        verdict = "data_fault" if p >= 0.75 else "normal" if p <= 0.25 else "unsure"
        if self.mode == "stubborn" or (self.mode == "disobedient" and not corrected):
            verdict = "unsure" if verdict != "unsure" else "normal"
        fault = verdict == "data_fault"
        expl = "mock explanation based on the ML tool output."
        if self.mode == "hallucinate" and not corrected:
            expl = "The readings are frozen at a flat value and the signal shows clipping at its maximum."
        return json.dumps({
            "action": "final",
            "report": {
                "verdict": verdict,
                "fault_type": res["predicted_fault_type"] if fault else None,
                "confidence": round(abs(p - 0.5) * 2, 3),
                "evidence": [{"tool": "ml_fault_probability", "field": "fault_probability", "value": shown}],
                "explanation": expl,
                "next_step": "human review" if verdict == "unsure" else ("review the flagged window" if fault else "no action"),
            },
        })


def extract_json(text):
    """Return the first complete JSON object found in `text`, or None."""
    if not isinstance(text, str):
        return None
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except Exception:
                            break
        start = text.find("{", start + 1)
    return None
