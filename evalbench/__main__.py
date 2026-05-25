"""python -m evalbench run   --out runs/candidate.json
   python -m evalbench gate  --baseline baselines/main.json --candidate runs/candidate.json
   python -m evalbench calibrate --labels data/human_labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import CaseResult, ClaimsGrader, FactsGrader, Gate, Grade, LengthGrader, RunReport, cohens_kappa, compare, load_cases, run, spearman

GATES = [Gate("pass_rate", 0.03), Gate("facts", 0.02, hard_floor=0.95), Gate("claims", 0.0, hard_floor=1.0), Gate("judge", 0.03)]


def load_report(path: str) -> RunReport:
    d = json.loads(Path(path).read_text())
    return RunReport(d["name"], [CaseResult(r["case_id"], r["output"], [Grade(**g) for g in r["grades"]]) for r in d["results"]], d.get("meta", {}))


def main(argv=None) -> int:
    p = argparse.ArgumentParser("evalbench")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--data", default="data/product_descriptions.jsonl"); r.add_argument("--out", required=True)
    g = sub.add_parser("gate"); g.add_argument("--baseline", required=True); g.add_argument("--candidate", required=True)
    c = sub.add_parser("calibrate"); c.add_argument("--labels", required=True); c.add_argument("--data", default="data/product_descriptions.jsonl")
    a = p.parse_args(argv)

    if a.cmd == "run":
        import anthropic
        from .judge import ClaudeJudge
        from .sut import PROMPT_VERSION, make_generator
        client = anthropic.Anthropic()
        rep = run("candidate", load_cases(Path(a.data)), make_generator(client),
                  [LengthGrader(), FactsGrader(), ClaimsGrader(), ClaudeJudge(client)], {"prompt_version": PROMPT_VERSION})
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(rep.to_json())
        print(json.dumps(rep.summary()))
        return 0

    if a.cmd == "gate":
        ok, lines = compare(load_report(a.baseline), load_report(a.candidate), GATES)
        print("\n".join(lines))
        print("GATE PASSED" if ok else "GATE FAILED")
        return 0 if ok else 1

    if a.cmd == "calibrate":
        import anthropic
        from .judge import ClaudeJudge
        judge = ClaudeJudge(anthropic.Anthropic())
        cases = {c.id: c for c in load_cases(Path(a.data))}
        human, model = [], []
        for line in Path(a.labels).read_text().splitlines():
            row = json.loads(line)
            v = judge.verdict(cases[row["case_id"]], row["output"])
            if v:
                human.append(row["accuracy"]); model.append(v.accuracy)
        print(json.dumps({"n": len(human), "kappa_accuracy": round(cohens_kappa(human, model), 3), "spearman": round(spearman(human, model), 3)}))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
