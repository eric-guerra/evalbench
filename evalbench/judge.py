"""LLM-as-judge grader using Claude structured outputs.

Design choices that make judge scores usable:
  * a written rubric with anchored levels (what a 1, 3 and 5 look like), not "rate 1-10"
  * the judge reasons first, then scores (rationale field before score in the schema)
  * structured output so parsing never fails
  * the judge sees the product facts, so "accurate" is checkable rather than vibes
  * pairwise mode runs both orders and only counts a win if it survives the swap (position bias)
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .core import Case, Grade

JUDGE_MODEL = "claude-opus-5-5"

RUBRIC = """You are grading a product description written for an online home-goods store.

Score each criterion from 1 to 5:

accuracy - Every factual statement is supported by the product facts. 5: nothing unsupported. \
3: one minor embellishment (e.g. an unstated color shade). 1: a material claim the facts contradict or don't support \
(e.g. "waterproof" when the facts say water-resistant).

usefulness - Helps a shopper decide. 5: covers what the item is, who it's for, key specs (size, material, care) \
in plain language. 3: generic but correct. 1: mostly filler or marketing fluff.

tone - Matches a warm, practical brand voice. 5: friendly and specific, no hype. 3: bland or slightly salesy. \
1: pushy, all-caps, or exaggerated superlatives.

Explain your reasoning briefly before giving the scores."""


class Verdict(BaseModel):
    rationale: str = Field(description="Two or three sentences, written before deciding the scores")
    accuracy: int = Field(ge=1, le=5)
    usefulness: int = Field(ge=1, le=5)
    tone: int = Field(ge=1, le=5)


class PairVerdict(BaseModel):
    rationale: str
    winner: Literal["A", "B", "tie"]


def _facts(case: Case) -> str:
    return "\n".join(f"- {k}: {v}" for k, v in case.input.items())


class ClaudeJudge:
    name = "judge"

    def __init__(self, client, model: str = JUDGE_MODEL):
        self.client, self.model = client, model

    def _parse(self, prompt: str, schema):
        r = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=4000,
            system=RUBRIC,
            output_format=schema,
            output_config={"effort": "low"},  # grading a short text: low effort keeps cost per case small
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if r.stop_reason == "refusal" or r.parsed_output is None:
            return None
        return r.parsed_output

    def verdict(self, case: Case, output: str) -> Verdict | None:
        return self._parse(f"Product facts:\n{_facts(case)}\n\nDescription to grade:\n<description>\n{output}\n</description>", Verdict)

    def grade(self, case: Case, output: str) -> Grade:
        v = self.verdict(case, output)
        if v is None:
            return Grade(self.name, 0.0, False, "judge returned no verdict")
        score = ((v.accuracy - 1) * 2 + (v.usefulness - 1) + (v.tone - 1)) / 16  # accuracy weighted 2x, normalised 0..1
        return Grade(self.name, round(score, 4), v.accuracy >= 4 and min(v.usefulness, v.tone) >= 3,
                     f"acc={v.accuracy} use={v.usefulness} tone={v.tone}: {v.rationale}")

    def pairwise(self, case: Case, a: str, b: str) -> Literal["A", "B", "tie"]:
        """Ask twice with the order swapped; disagreement between orders counts as a tie."""
        def ask(first, second):
            v = self._parse(f"Product facts:\n{_facts(case)}\n\nWhich description is better overall?\n\n"
                            f"<description_A>\n{first}\n</description_A>\n\n<description_B>\n{second}\n</description_B>", PairVerdict)
            return v.winner if v else "tie"
        w1 = ask(a, b)
        w2 = {"A": "B", "B": "A", "tie": "tie"}[ask(b, a)]
        return w1 if w1 == w2 else "tie"
