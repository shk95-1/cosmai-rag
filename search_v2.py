"""
search_v2.py — 시현님 리포 모듈로 교체한 검색 오케스트레이터
"""
import os
import sys
from pathlib import Path

REPO = os.environ.get("SEARCH_REPO", ".")  # 이 파일과 시현님 모듈이 같은 폴더에 있다고 가정
sys.path.insert(0, REPO)
os.chdir(REPO)  # bm25.py 가 'seeds/...' 상대경로로 사전을 찾는다 - cwd 를 맞춰야 함

import bm25
from bm25 import by_source, tokenize_query
from hybrid import Vectors
from vector_threshold import df_gate

COMMON = Path("common")
CHUNKS = [Path("reports/chunks_ingredient_mfds.csv"), Path("reports/chunks_commerce.csv")]
ALL_CHUNKS = CHUNKS + [Path("reports/chunks_youtube.csv")]  # 벡터는 유튜브도 포함
VECTORS = Path(".cache/vectors/e5all")


class SearchEngineV2:
    def __init__(self):
        self.index, self.origin = bm25.build(COMMON, None, Path(".cache/bm25"), CHUNKS)
        _ids, bodies, _o = bm25.load_documents(COMMON, None)
        more_ids, more_bodies, more_origin = bm25.load_chunks(ALL_CHUNKS)  # 벡터용 - 유튜브 포함
        all_ids = _ids + more_ids
        all_bodies = bodies + more_bodies
        self.body = dict(zip(all_ids, all_bodies))
        self.full_origin = {**self.origin, **more_origin}  # 벡터 결과의 소스 라벨용
        self.vectors = Vectors(VECTORS)

    def search_bm25(self, query: str, k: int = 10):
        return [
            {"doc_id": d, "score": s, "source": self.origin.get(d, "?"),
             "text": self.body.get(d, "")}
            for d, s in self.index.search(query, k)
        ]

    def search_bm25_per_source(self, query: str, k: int = 5):
        found = by_source(self.index, self.origin, query, k)
        return {
            src: [{"doc_id": d, "score": s, "text": self.body.get(d, "")}
                 for d, s in hits]
            for src, hits in found.items()
        }

    def search_vector(self, query: str, k: int = 10):
        ok, why = df_gate(query, self.index)
        if not ok:
            return {"results": [], "gated": True, "reason": why}
        hits = self.vectors.search(query, k)
        return {
            "results": [
                {"doc_id": d, "score": s, "source": self.full_origin.get(d, "?"),
                 "text": self.body.get(d, "")}
                for d, s in hits
            ],
            "gated": False, "reason": why,
        }


def demo():
    eng = SearchEngineV2()

    print("=" * 70)
    print("1) BM25 - 등록번호 (전역, 정확 일치 검증)")
    for r in eng.search_bm25("보고번호 2018008612", k=3):
        print(f"  {r['score']:.2f} [{r['source']}] {r['text'][:60]}")

    print()
    print("=" * 70)
    print("2) BM25 소스별 (by_source) - '백탁 관련해서 소비자들이'")
    found = eng.search_bm25_per_source("백탁 관련해서 소비자들이", k=2)
    for src, hits in found.items():
        print(f"  [{src}]")
        for h in hits:
            print(f"    {h['score']:.2f}  {h['text'][:55]}")

    print()
    print("=" * 70)
    print("3) 벡터 + df_gate - 존재하지 않는 성분")
    r = eng.search_vector("크소나이드", k=5)
    print(f"  gated={r['gated']} | {r['reason']}")
    print(f"  결과: {len(r['results'])}건")

    print()
    print("4) 벡터 + df_gate - 진짜 자연어 표현")
    r = eng.search_vector("하얗게", k=3)
    print(f"  gated={r['gated']} | {r['reason']}")
    for x in r["results"]:
        print(f"    {x['score']:.3f}  {x['text'][:55]}")


if __name__ == "__main__":
    demo()
