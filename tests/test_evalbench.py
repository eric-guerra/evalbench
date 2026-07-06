import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from evalbench.__main__ import GATES, main
from evalbench.core import Case, ClaimsGrader, FactsGrader, Gate, LengthGrader, cohens_kappa, compare, load_cases, paired_bootstrap, run, spearman
from evalbench.judge import JUDGE_MODEL, ClaudeJudge, PairVerdict, Verdict
from evalbench.sut import make_generator

DATA = Path(__file__).parent.parent / "data" / "product_descriptions.jsonl"
CASES = load_cases(DATA)


def good_gen(facts):
    return (f"Meet the {facts['name']}, made from {facts['material']} and sized at {facts['size']}. "
            f"It brings an easy, lived-in warmth to everyday rooms and pairs well with what you already own. "
            f"Care is simple: {facts['care']}. Features: {facts['features']}. A practical piece you can use daily "
            f"without fuss, built to look good for years.")


def sloppy_gen(facts):  # drops the size on every third product, and over-claims on outdoor items
    out = good_gen(facts)
    if "Outdoor" in facts["name"]:
        out = out.replace("Features:", "Fully waterproof. Features:")
    if len(facts["name"]) % 3 == 0:  # deterministic (str hash() is randomized per process)
        out = out.replace(f" and sized at {facts['size']}", "")
    return out


GRADERS = [LengthGrader(), FactsGrader(), ClaimsGrader()]


def test_dataset_is_well_formed():
    assert len(CASES) == 24 and len({c.id for c in CASES}) == 24
    assert all(c.must_mention for c in CASES)


def test_deterministic_graders():
    c = Case("x", {}, must_mention=("cotton", "50 x 60 in"), must_not_claim=("waterproof",))
    assert FactsGrader().grade(c, "Soft COTTON throw, 50  x 60 in.").passed
    assert FactsGrader().grade(c, "Soft cotton throw.").score == 0.5
    assert not ClaimsGrader().grade(c, "Totally waterproof and guaranteed!").passed
    assert ClaimsGrader().grade(c, "Water-resistant finish.").passed  # 'water-resistant' ≠ 'waterproof'
    assert not LengthGrader(min_words=5).grade(c, "Too short").passed


def test_good_generator_passes_and_sloppy_one_is_caught():
    good = run("good", CASES, good_gen, GRADERS)
    sloppy = run("sloppy", CASES, sloppy_gen, GRADERS)
    assert good.pass_rate() == 1.0
    assert sloppy.metric("claims") < 1.0 and sloppy.metric("facts") < 1.0
    ok, lines = compare(good, sloppy, [Gate("pass_rate", 0.03), Gate("facts", 0.02, hard_floor=0.95), Gate("claims", 0.0, hard_floor=1.0)])
    assert not ok and any(l.startswith("FAIL claims") for l in lines)
    ok2, _ = compare(good, run("good2", CASES, good_gen, GRADERS), [Gate("pass_rate", 0.03)])
    assert ok2


def test_bootstrap_ci_brackets_the_true_difference():
    a = [1.0] * 50
    b = [1.0] * 45 + [0.0] * 5
    d, lo, hi = paired_bootstrap(a, b)
    assert d == pytest.approx(-0.1) and lo < d < hi and hi < 0


def test_agreement_metrics():
    assert cohens_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    assert abs(cohens_kappa([1, 1, 2, 2], [1, 2, 1, 2])) < 1e-9
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


class FakeClaude:
    """Returns scripted verdicts; records requests."""
    def __init__(self, verdicts):
        self.verdicts, self.calls = list(verdicts), []
        self.beta = NS(messages=NS(parse=self._parse))
        self.messages = NS(create=self._create)

    def _parse(self, **kw):
        self.calls.append(kw)
        return NS(parsed_output=self.verdicts.pop(0), stop_reason="end_turn")

    def _create(self, **kw):
        self.calls.append(kw)
        return NS(stop_reason="end_turn", content=[NS(type="text", text="  A sage cotton throw.  ")])


def test_judge_request_and_scoring():
    fake = FakeClaude([Verdict(rationale="Accurate and helpful.", accuracy=5, usefulness=4, tone=4)])
    g = ClaudeJudge(fake).grade(CASES[0], "desc")
    req = fake.calls[0]
    assert req["model"] == JUDGE_MODEL == "claude-opus-5-5" and req["output_format"] is Verdict
    assert req["fallbacks"] == "default" and req["output_config"] == {"effort": "low"}
    assert "material: cotton" in req["messages"][0]["content"]
    assert g.passed and g.score == pytest.approx((8 + 3 + 3) / 16)


def test_inaccurate_description_fails_judge_even_if_well_written():
    fake = FakeClaude([Verdict(rationale="Claims waterproof.", accuracy=2, usefulness=5, tone=5)])
    assert not ClaudeJudge(fake).grade(CASES[0], "desc").passed


def test_pairwise_requires_the_win_to_survive_order_swap():
    consistent = FakeClaude([PairVerdict(rationale="", winner="A"), PairVerdict(rationale="", winner="B")])
    assert ClaudeJudge(consistent).pairwise(CASES[0], "x", "y") == "A"
    position_biased = FakeClaude([PairVerdict(rationale="", winner="A"), PairVerdict(rationale="", winner="A")])
    assert ClaudeJudge(position_biased).pairwise(CASES[0], "x", "y") == "tie"


def test_generator_request_shape():
    fake = FakeClaude([])
    out = make_generator(fake)({"name": "Throw", "material": "cotton"})
    assert out == "A sage cotton throw."
    assert fake.calls[0]["model"] == "claude-opus-5-5" and "thinking" not in fake.calls[0]


def test_cli_gate(tmp_path, capsys):
    base = run("base", CASES, good_gen, GRADERS)
    cand = run("cand", CASES, sloppy_gen, GRADERS)
    (tmp_path / "b.json").write_text(base.to_json())
    (tmp_path / "c.json").write_text(cand.to_json())
    gates_no_judge = [g for g in GATES if g.metric != "judge"]
    import evalbench.__main__ as m
    m.GATES[:] = gates_no_judge
    assert main(["gate", "--baseline", str(tmp_path / "b.json"), "--candidate", str(tmp_path / "c.json")]) == 1
    assert "GATE FAILED" in capsys.readouterr().out
    assert main(["gate", "--baseline", str(tmp_path / "b.json"), "--candidate", str(tmp_path / "b.json")]) == 0
