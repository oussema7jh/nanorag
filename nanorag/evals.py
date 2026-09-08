"""Golden-set evaluation for retrieval and grounded answering.

Evals are what separate a demo from an engineered system. This module runs
a labelled question set against the pipeline and reports:

* **recall@k** — did a chunk from the expected document make the top-k?
* **MRR**     — mean reciprocal rank of the first correct chunk
* **citation rate** — did the generated answer cite a correct passage?

Run: ``nanorag eval --golden evals/golden.jsonl``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from .pipeline import RAGPipeline


@dataclass
class GoldenCase:
    question: str
    expected_doc: str

    @classmethod
    def from_dict(cls, d: dict) -> "GoldenCase":
        return cls(question=d["question"], expected_doc=d["expected_doc"])


@dataclass
class CaseResult:
    case: GoldenCase
    first_rank: Optional[int]  # rank of first passage from the expected doc
    cited_correctly: bool
    top_sources: List[str]


@dataclass
class EvalReport:
    results: List[CaseResult] = field(default_factory=list)

    @property
    def recall_at_k(self) -> float:
        if not self.results:
            return 0.0
        hits = sum(1 for r in self.results if r.first_rank is not None)
        return hits / len(self.results)

    @property
    def mrr(self) -> float:
        if not self.results:
            return 0.0
        total = sum(1.0 / r.first_rank for r in self.results if r.first_rank)
        return total / len(self.results)

    @property
    def citation_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.cited_correctly) / len(self.results)

    def table(self) -> str:
        header = f"{'QUESTION':<44} {'RANK':>4} {'CITED':>6}"
        lines = [header, "-" * len(header)]
        for r in self.results:
            q = (r.case.question[:41] + "...") if len(r.case.question) > 44 else r.case.question
            rank = str(r.first_rank) if r.first_rank else "-"
            cited = "yes" if r.cited_correctly else "no"
            lines.append(f"{q:<44} {rank:>4} {cited:>6}")
        lines.append("-" * len(header))
        lines.append(
            f"recall@5: {self.recall_at_k:.0%}   "
            f"MRR: {self.mrr:.3f}   "
            f"citation rate: {self.citation_rate:.0%}   "
            f"({len(self.results)} cases)"
        )
        return "\n".join(lines)


def load_golden(path: str) -> List[GoldenCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(GoldenCase.from_dict(json.loads(line)))
    return cases


def _matches(source: str, expected: str) -> bool:
    return Path(source).name == Path(expected).name


def evaluate(pipeline: RAGPipeline, golden: Sequence[GoldenCase], k: int = 5) -> EvalReport:
    report = EvalReport()
    for case in golden:
        passages = pipeline.retrieve(case.question, k)
        first_rank = None
        for p in passages:
            if _matches(p.source or p.chunk.doc_id, case.expected_doc):
                first_rank = p.rank
                break

        cited = False
        answer = pipeline.ask(case.question, k)
        for p in passages:
            if _matches(p.source or p.chunk.doc_id, case.expected_doc) and f"[{p.rank}]" in answer.text:
                cited = True
                break

        report.results.append(
            CaseResult(
                case=case,
                first_rank=first_rank,
                cited_correctly=cited,
                top_sources=[(p.source or p.chunk.doc_id) for p in passages[:3]],
            )
        )
    return report
