"""System under test: the product-description generator (a prompt + Claude). Versions of the prompt
are what the eval compares; `PROMPT_VERSION` is recorded in every report."""
from __future__ import annotations

GEN_MODEL = "claude-opus-5-5"
PROMPT_VERSION = "v3"

SYSTEM = """You write product descriptions for a home-goods store. Write 50-90 words in a warm, practical voice.
Use only the facts provided: state the material and size exactly as given, mention care instructions, \
and never claim properties that aren't in the facts (for example, don't call something waterproof if it is \
water-resistant). No superlatives, no exclamation marks."""


def make_generator(client, model: str = GEN_MODEL):
    def generate(facts: dict) -> str:
        r = client.messages.create(
            model=model,
            max_tokens=2000,
            system=SYSTEM,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": "\n".join(f"{k}: {v}" for k, v in facts.items())}],
        )
        if r.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in r.content if b.type == "text").strip()
    return generate
