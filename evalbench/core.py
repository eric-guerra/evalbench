"""EvalBench: regression evals for LLM features (here: product-description generation for a catalog).

A run = dataset × system-under-test → outputs → graders → scores → compared to a stored baseline.
Graders are layered cheapest-first:
  1. deterministic checks (length, required facts, banned claims, format) — free, exact
  2. an LLM judge with a written rubric and structured output — for qualities code can't check
The judge itself is calibrated against human labels before its scores are trusted.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol


@dataclass(frozen=True)
class Case:
    id: str
    input: dict  # product facts given to the generator
    must_mention: tuple[str, ...] = ()
    must_not_claim: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


def load_cases(path: Path) -> list[Case]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(Case(d["id"], d["input"], tuple(d.get("must_mention", ())), tuple(d.get("must_not_claim", ())), tuple(d.get("tags", ()))))
    return out


@dataclass
class Grade:
    name: str
    score: float  # 0..1
    passed: bool
    detail: str = ""


class Grader(Protocol):
    name: str
    def grade(self, case: Case, output: str) -> Grade: ...


# ---------------------------------------------------------------- deterministic graders

@dataclass
class LengthGrader:
    min_words: int = 40
    max_words: int = 120
    name: str = "length"

    def grade(self, case, output):
        n = len(output.split())
        ok = self.min_words <= n <= self.max_words
        return Grade(self.name, 1.0 if ok else 0.0, ok, f"{n} words")


@dataclass
class FactsGrader:
    """Every required fact appears (case-insensitive, whitespace-tolerant)."""
    name: str = "facts"

    def grade(self, case, output):
        norm = re.sub(r"\s+", " ", output.lower())
        missing = [f for f in case.must_mention if re.sub(r"\s+", " ", f.lower()) not in norm]
        score = 1.0 - len(missing) / len(case.must_mention) if case.must_mention else 1.0
        return Grade(self.name, score, not missing, f"missing: {missing}" if missing else "")


@dataclass
class ClaimsGrader:
    """Compliance: no unsupported claims (e.g. 'waterproof' on a water-resistant item, health claims)."""
    global_banned: tuple[str, ...] = ("clinically proven", "guaranteed", "cures", "100% safe", "best in the world")
    name: str = "claims"

    def grade(self, case, output):
        low = output.lower()
        hits = [c for c in (*self.global_banned, *case.must_not_claim) if re.search(rf"\b{re.escape(c.lower())}\b", low)]
        return Grade(self.name, 0.0 if hits else 1.0, not hits, f"banned: {hits}" if hits else "")


# ---------------------------------------------------------------- run + compare

@dataclass
class CaseResult:
    case_id: str
    output: str
    grades: list[Grade]
    tags: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.grades)


@dataclass
class RunReport:
    name: str
    results: list[CaseResult]
    meta: dict = field(default_factory=dict)

    def metric(self, grader: str) -> float:
        xs = [g.score for r in self.results for g in r.grades if g.name == grader]
        return sum(xs) / len(xs) if xs else float("nan")

    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results)

    def summary(self) -> dict:
        graders = sorted({g.name for r in self.results for g in r.grades})
        return {"pass_rate": round(self.pass_rate(), 4), **{g: round(self.metric(g), 4) for g in graders}, "n": len(self.results)}

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "meta": self.meta, "summary": self.summary(),
                           "results": [{"case_id": r.case_id, "passed": r.passed, "output": r.output,
                                        "grades": [asdict(g) for g in r.grades]} for r in self.results]}, indent=2)


def run(name: str, cases: list[Case], generate: Callable[[dict], str], graders: list[Grader], meta: dict | None = None) -> RunReport:
    results = []
    for c in cases:
        out = generate(c.input)
        results.append(CaseResult(c.id, out, [g.grade(c, out) for g in graders], c.tags))
    return RunReport(name, results, meta or {})


def paired_bootstrap(a: list[float], b: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean(b - a) with a 95% CI from a paired bootstrap over cases."""
    import random
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(diffs) / n, boots[int(0.025 * iters)], boots[int(0.975 * iters) - 1]


@dataclass
class Gate:
    """CI decision: fail only on a regression that is both larger than `tolerance` and statistically real."""
    metric: str
    tolerance: float = 0.02
    hard_floor: float | None = None


def compare(baseline: RunReport, candidate: RunReport, gates: list[Gate]) -> tuple[bool, list[str]]:
    by_id = {r.case_id: r for r in baseline.results}
    lines, ok = [], True
    for gate in gates:
        def score(r: CaseResult):
            if gate.metric == "pass_rate":
                return 1.0 if r.passed else 0.0
            return next(g.score for g in r.grades if g.name == gate.metric)
        pairs = [(score(by_id[r.case_id]), score(r)) for r in candidate.results if r.case_id in by_id]
        a, b = [p[0] for p in pairs], [p[1] for p in pairs]
        delta, lo, hi = paired_bootstrap(a, b)
        regressed = delta < -gate.tolerance and hi < 0
        below_floor = gate.hard_floor is not None and (sum(b) / len(b)) < gate.hard_floor
        status = "FAIL" if regressed or below_floor else "ok"
        ok &= status == "ok"
        lines.append(f"{status:4} {gate.metric:12} baseline {sum(a)/len(a):.3f} → {sum(b)/len(b):.3f}  Δ {delta:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]"
                     + ("  (below floor)" if below_floor else ""))
    return ok, lines


# ---------------------------------------------------------------- judge calibration

def cohens_kappa(a: list[int], b: list[int]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return cov / den if den else 0.0
