# EvalBench

Portfolio reference implementation of a 2025 project, built to show a regression-eval harness for an LLM feature: AI-written product descriptions for a home-goods catalog. Prompt and model changes ship only when the evals say they didn't make things worse.

**Stack:** Python 3.11, pytest, Anthropic Python SDK (Claude Opus 5.5 as generator and judge, with structured outputs), Pydantic, GitHub Actions.

## Layers of grading (cheapest first)
| Grader | Checks | Cost |
|---|---|---|
| `LengthGrader` | 40–120 words | free |
| `FactsGrader` | material and size appear exactly as given | free |
| `ClaimsGrader` | no banned claims: global ("clinically proven", "guaranteed") and per item ("waterproof" on water-resistant goods, "dishwasher safe" on bamboo) | free |
| `ClaudeJudge` | accuracy, usefulness and tone on an **anchored rubric** (what a 1, 3 and 5 look like); accuracy weighted 2× | 1 call/case |

**Judge design** (`evalbench/judge.py`):
- Structured output via `client.beta.messages.parse(output_format=Verdict)`, so parsing never fails.
- The rationale field comes *before* the scores, so the judge reasons before it decides.
- The judge sees the product facts, which makes "accurate" checkable.
- Low effort keeps the per-case cost small.
- Server-side fallbacks are on, and a refusal counts as a failed verdict.
- **Pairwise mode** asks twice with the order swapped and counts a win only if it survives the swap, which removes position bias.

**Calibrate before trusting the judge:** `python -m evalbench calibrate --labels data/human_labels.jsonl` reports Cohen's κ and Spearman ρ between the judge and human accuracy labels.

## The gate
`python -m evalbench gate --baseline baselines/main.json --candidate runs/candidate.json`
- Compares the candidate with the baseline **per case** using a **paired bootstrap** with a 95% CI.
- A metric fails only when the drop is larger than its tolerance *and* the CI excludes zero. Noise doesn't block merges; real regressions do.
- Hard floors are absolute: facts ≥ 0.95, and claims = 1.0, since compliance isn't negotiable.

Real output from the test suite, comparing the good generator with the "sloppy" one, which drops sizes and calls outdoor cushions waterproof:
```
FAIL pass_rate    baseline 1.000 → 0.625  Δ -0.375 [95% CI -0.583, -0.167]
FAIL facts        baseline 1.000 → 0.833  Δ -0.167 [95% CI -0.271, -0.083]  (below floor)
FAIL claims       baseline 1.000 → 0.875  Δ -0.125 [95% CI -0.250, +0.000]  (below floor)
GATE FAILED
```
The claims CI touches zero (only 3 outdoor items), so the floor is what catches it. That's why compliance metrics get a hard floor rather than relying on significance alone.

## CI (`.github/workflows/evals.yml`)
- PRs that touch the prompt, data or judge run the unit tests, then the live eval, then the gate.
- The gate result is posted as a sticky PR comment, and a failing gate blocks the merge.
- Merges to main refresh `baselines/main.json`.

## Test
```bash
pip install -r requirements.txt
pytest -q   # 10 tests, no API calls: graders, a "sloppy" generator caught by the gate, bootstrap CI,
            # κ and ρ, judge request and scoring, swap-consistent pairwise, CLI exit codes
```
