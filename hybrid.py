#!/usr/bin/env python3
"""BM25 와 벡터를 합친다(RRF). 그리고 셋을 같은 기준으로 채점할 수 있게 만든다.

**핵심은 합치는 방법이 아니라 비교입니다.** 하이브리드가 BM25 단독보다 나은지
확인하지 않고 붙이면, 좋아졌다고 믿을 근거가 없다. 그래서 이 파일은 검색기 세 개를
같은 인터페이스로 내놓는다 — `retrieval_eval.py` 가 `--engine` 으로 골라 쓴다.

    bm25     어휘. 성분명·SPF 수치처럼 정확 일치가 정답인 것에 강하다
    vector   의미. `하얗게 떠서 싫다` 처럼 이름 없는 표현에 강하다
    hybrid   RRF 로 두 순위를 합친다

RRF(reciprocal rank fusion)를 쓰는 이유. 두 검색기의 **점수 스케일이 다르다** —
BM25 는 11.83 같은 값이고 코사인은 0~1 이다. 점수를 정규화해 더하면 그 정규화
방식이 또 하나의 손잡이가 된다. RRF 는 **순위만** 쓰므로 스케일 문제가 없다.

    score(d) = Σ 1 / (k + rank_i(d))        k = 60 (관행값)

k=60 은 흔히 쓰는 값이고 우리 데이터로 다시 뽑지 않았다. 뽑으려면 자동 라벨
(`retrieval_eval --mode literal`)로 골라야 하고, 골든셋으로 고르면 그 숫자는
성능이 아니라 우리가 맞춘 결과가 된다.

**채택 기준은 미리 정해 뒀다.** `heldout` 모드에서 BM25 는 구조적으로 0.000 이다
(정답을 "질의 토큰이 없는 문서"로 잡았으므로). 벡터가 **0을 넘으면** 그게 임베딩이
기여한 몫이고, 못 넘으면 붙일 이유가 없다.

사용법:
    python retrieval_eval.py --engine bm25    --mode heldout
    python retrieval_eval.py --engine vector  --mode heldout
    python retrieval_eval.py --engine hybrid  --mode heldout
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import bm25

csv.field_size_limit(10 ** 8)

RRF_K = 60          # 순위 융합 상수. 관행값이고 우리 데이터로 다시 뽑지 않았다
VECTORS = Path(".cache/vectors/e5base")


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """순위 목록 여러 개를 합친다. 점수가 아니라 **순위만** 쓴다.

    한쪽에만 있는 문서도 들어온다 — 그게 하이브리드의 요점이다. BM25 가 못 찾은
    것을 벡터가 찾고, 그 반대도 있다.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, 1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class Vectors:
    """벡터 검색. L2 정규화된 행렬이면 코사인이 내적이라 행렬곱 한 번이다."""

    def __init__(self, base: Path):
        manifest_path = base.with_suffix(".manifest.json")
        if not manifest_path.exists():
            raise SystemExit(
                f"벡터가 없다: {base}.npy\n"
                f"먼저 인코딩한다: python encode_chunks.py --out {base}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.matrix = np.load(base.with_suffix(".npy"))
        with base.with_suffix(".ids.csv").open(encoding="utf-8", newline="") as handle:
            self.ids = [row["chunk_id"] for row in csv.DictReader(handle)]
        if len(self.ids) != len(self.matrix):
            raise SystemExit(
                f"벡터 {len(self.matrix)} 와 id {len(self.ids)} 개수가 다르다. "
                f"인코딩을 다시 해야 한다")
        if not self.manifest.get("l2_normalized"):
            raise SystemExit("L2 정규화가 안 된 벡터다. 내적을 코사인으로 쓸 수 없다")
        self._model = None

    def encode_query(self, query: str) -> np.ndarray:
        """질의에는 **문서와 다른 프리픽스**를 붙인다. 매니페스트에 적힌 것을 쓴다."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise SystemExit("sentence-transformers 가 없다. pip install 이 필요하다")
            self._model = SentenceTransformer(self.manifest["model"])
        prefix = self.manifest.get("query_prefix", "query: ")
        vector = self._model.encode([prefix + query], convert_to_numpy=True)
        vector = vector.astype(np.float32)
        norm = np.linalg.norm(vector)
        return vector / (norm or 1.0)

    def search(self, query: str, k: int | None = 10) -> list[tuple[str, float]]:
        similarity = (self.matrix @ self.encode_query(query).T).ravel()
        take = len(similarity) if k is None else min(k, len(similarity))
        # argpartition 은 상위 k 만 고르므로 30만 행에서도 정렬보다 훨씬 빠르다
        top = np.argpartition(-similarity, take - 1)[:take]
        top = top[np.argsort(-similarity[top])]
        return [(self.ids[i], round(float(similarity[i]), 4)) for i in top]


def make_engine(name: str, common: Path, chunks: list[Path],
                vectors: Path = VECTORS, no_cache: bool = False):
    """(검색 함수, 소스 사전). 검색 함수는 `(query, k) -> [(doc_id, score)]`."""
    if name == "vector":
        store = Vectors(vectors)
        origin = _origin_from_chunks(chunks)
        return store.search, origin

    index, origin = bm25.build(common, None,
                               None if no_cache else Path(".cache/bm25"), chunks)
    if name == "bm25":
        return index.search, origin

    store = Vectors(vectors)

    def fused(query: str, k: int | None = 10) -> list[tuple[str, float]]:
        # 각 검색기에서 넉넉히 받아 합친다. k 만 받으면 한쪽에만 있는 문서를 놓친다
        depth = 200 if k is None else max(k * 10, 100)
        lexical = [d for d, _s in index.search(query, depth)]
        dense = [d for d, _s in store.search(query, depth)]
        merged = rrf([lexical, dense])
        return merged if k is None else merged[:k]

    return fused, origin


def _origin_from_chunks(chunks: list[Path]) -> dict[str, str]:
    origin = {}
    for path in chunks:
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                origin[row["chunk_id"]] = row["source"]
    return origin


def demo() -> None:
    # 두 순위에 다 높으면 합산 점수가 가장 높다
    merged = dict(rrf([["a", "b", "c"], ["a", "c", "b"]]))
    assert merged["a"] > merged["b"] and merged["a"] > merged["c"]
    # 한쪽에만 있는 문서도 들어온다 — 그게 하이브리드의 요점이다
    merged = dict(rrf([["a"], ["z"]]))
    assert set(merged) == {"a", "z"} and merged["a"] == merged["z"]
    # 순위만 쓴다. 점수 스케일이 달라도 결과가 안 바뀐다
    assert rrf([["a", "b"]]) == [("a", 1 / 61), ("b", 1 / 62)]
    # 1위가 2위보다 항상 크다
    ranked = rrf([[f"d{i}" for i in range(50)]])
    assert ranked[0][1] > ranked[1][1] > ranked[-1][1]

    # 정규화된 벡터에서 내적 = 코사인
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    similarity = (matrix @ query.T).ravel()
    assert similarity.argmax() == 0 and abs(similarity[2] - 0.7071) < 1e-3
    print("demo ok")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine", choices=["bm25", "vector", "hybrid"], default="hybrid")
    p.add_argument("--query")
    p.add_argument("--common", type=Path, default=Path("common"))
    p.add_argument("--chunks", action="append", type=Path, default=[])
    p.add_argument("--vectors", type=Path, default=VECTORS)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()
    if a.demo:
        demo()
        return 0
    if not a.query:
        p.error("--query 를 주거나 --demo 를 쓴다")
    chunks = a.chunks or [Path("reports/chunks_ingredient_mfds.csv")]
    search, origin = make_engine(a.engine, a.common, chunks, a.vectors)
    for rank, (doc_id, score) in enumerate(search(a.query, a.top), 1):
        print(f"{rank:>3}. {score:>8.4f}  {origin.get(doc_id, '?'):<18}{doc_id[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
