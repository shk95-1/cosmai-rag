"""
generate.py — LLM 답변 생성 레이어 (search_v2 로 재연결)

router.py 로 질의 유형을 판단하되, 실제 검색 실행은 search_v2.SearchEngineV2
(= 시현님의 bm25.py/hybrid.py/vector_threshold.py)를 쓴다.
temporal_filter(시점 필터)만 시현님 모듈에 대응하는 게 없어서 원래 로직을 유지한다
(dim_mfds_item.report_date 직접 조회 - 텍스트 검색이 아니라 날짜 메타데이터 필터라
BM25/벡터 어느 쪽 소관도 아니다).

바뀐 것 (router.py 는 그대로, 백엔드만 교체):
  BM25 실행       -> search_v2.search_bm25 / search_bm25_per_source (시현님 코드)
  벡터 실행       -> search_v2.search_vector (df_gate 내장 - 코사인 임계값 아님)
  시점 필터       -> 기존 sqlite 직접 조회 유지
"""
import json
import os
from dataclasses import dataclass, field

from router import QueryRouter, RouteDecision
import search_v2

SYSTEM_PROMPT = """당신은 화장품 트렌드/성분 데이터 어시스턴트입니다. 아래 규칙을 반드시 지키세요.

1. 오직 제공된 근거(evidence)에 있는 내용만으로 답하세요. 근거에 없는 사실을
   추측하거나 만들어내지 마세요.
2. 모든 사실 주장 뒤에는 [출처: doc_id] 형식으로 근거를 표시하세요.
3. 근거가 부족하거나 없으면 "제공된 데이터로는 답할 수 없습니다"라고 명시하고,
   억지로 답하지 마세요.
4. 근거가 여러 출처(source)에서 왔으면 섞어서 결론 내지 말고 출처별로 구분해서
   제시하세요. BM25/벡터 점수는 서로 다른 검색 방식·코퍼스에서 나온 것이라
   직접 비교할 수 없습니다 - 순위 근거로 점수 수치를 언급하지 마세요.
5. 식약처 등록 데이터(mfds)는 "등록됨/보고됨" 사실만 말하고, 효과나 안전성을
   보증하는 것처럼 말하지 마세요 (보고=효과 입증 아님).
6. 성분과 소비자 반응 사이의 인과관계를 단정하지 마세요. "~라는 언급이
   있었다" 정도로만 서술하세요.
7. 시점 필터 결과(temporal_filter)는 "최근 N일간 등록된 품목 목록"이지
   "트렌드가 상승했다"는 뜻이 아닙니다. 판정을 만들어내지 마세요.
8. 벡터 검색 결과는 Hit@10 13% 수준으로 놓치는 경우가 훨씬 많고, 질의와
   의미상 무관해도 코사인 유사도가 높게 나올 수 있습니다(e5 임베딩의 알려진
   한계). 그래서 아래 "질의 토큰별 코퍼스 등장 빈도"를 반드시 먼저 확인하세요.
   - 질의의 핵심 명사(성분명·고유명사 등)의 df(등장 문서 수)가 0이면, 그
     단어는 코퍼스에 전혀 존재하지 않는다는 뜻입니다. 검색 결과가 나왔더라도
     그건 그 핵심 단어와 무관하게 우연히 비슷한 벡터를 가진 문서일 뿐입니다.
   - 이 경우 결과 텍스트를 실제로 읽고 질의의 핵심 단어·개념과 정말 관련
     있는지 직접 판단하세요. 관련 없으면 점수가 나왔어도 "제공된 데이터로는
     답할 수 없습니다"라고 답하세요. df가 0인 핵심 단어가 있다는 사실 자체도
     답변에서 언급하세요."""


@dataclass
class GeneratedAnswer:
    question: str
    route: str
    prompt: str
    evidence_count: int
    citations: list = field(default_factory=list)
    answer: str = ""
    executed: bool = False
    note: str = ""


class AnswerGenerator:
    def __init__(self, api_key: str | None = None):
        self.router = QueryRouter()
        self.engine = search_v2.SearchEngineV2()
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    # -------------------------------------------------------------
    def _format_evidence(self, route: str, payload) -> tuple[str, list]:
        citations = []

        if route == "temporal_filter":
            rows = payload["results"]
            if not rows:
                return "(해당 기간 내 등록된 품목 없음)", []
            lines = [f"기간: {payload['window']}",
                    f"전체 {payload['n_total_in_window']}건 중 상위 {len(rows)}건:"]
            for r in rows:
                cid = f"MFDS:{r['COSMETIC_REPORT_SEQ']}"
                lines.append(f"- [{cid}] {r['ITEM_NAME']} "
                            f"({r['ENTP_NAME']}, {r['report_date']})")
                citations.append(cid)
            return "\n".join(lines), citations

        if route == "multi_source":
            blocks = []
            for src, hits in payload.items():
                if not hits:
                    continue
                blocks.append(f"[출처: {src}]")
                for h in hits:
                    blocks.append(f"- [{h['doc_id']}] {h['text'][:200]}")
                    citations.append(h["doc_id"])
            if not blocks:
                return "(어느 출처에서도 관련 근거를 찾지 못함)", []
            return "\n".join(blocks), citations

        if route == "bm25":
            if not payload:
                return "(관련 근거를 찾지 못함)", []
            lines = [f"- [{h['doc_id']}] (출처:{h['source']}) {h['text'][:200]}"
                    for h in payload]
            citations = [h["doc_id"] for h in payload]
            return "\n".join(lines), citations

        if route == "vector":
            if payload.get("gated"):
                return f"(df_gate 차단: {payload['reason']})", []
            hits = payload["results"]
            if not hits:
                return "(관련 근거를 찾지 못함)", []
            lines = [f"- [{h['doc_id']}] (출처:{h['source']}) {h['text'][:200]}"
                    for h in hits]
            citations = [h["doc_id"] for h in hits]
            return "\n".join(lines), citations

        return "(알 수 없는 라우팅)", []

    def _token_df_report(self, query: str) -> str:
        """질의 토큰별 코퍼스 등장 빈도. LLM이 스스로 판단할 구체적 근거를 준다.
        '조심해라'는 두루뭉술한 지시보다, '크소나이드는 df=0' 같은 검증 가능한
        사실을 주는 게 훨씬 강하게 작동한다."""
        import bm25
        toks = bm25.tokenize_query(query)
        if not toks:
            return "(질의 토큰 없음 - df 판정 불가)"
        lines = []
        for t in sorted(set(toks)):
            df = len(self.engine.index.postings.get(t, ()))
            flag = " ← 코퍼스에 전혀 없음" if df == 0 else ""
            lines.append(f"  {t}: {df:,}건{flag}")
        return "질의 토큰별 코퍼스 등장 빈도:\n" + "\n".join(lines)

    # -------------------------------------------------------------
    def build_prompt(self, question: str, k: int = 8,
                     force_multi_source: bool = False) -> GeneratedAnswer:
        if force_multi_source:
            payload = self.engine.search_bm25_per_source(question, k=3)
            route = "multi_source"
        else:
            decision = self.router.classify(question)
            route = decision.route
            if route == "temporal_filter":
                payload = self._search_temporal(decision, k)
            elif route == "bm25":
                from router import DOMAIN_TOPIC_TERMS
                real_ingredients = [i for i in decision.matched_ingredients
                                    if i not in DOMAIN_TOPIC_TERMS]
                if real_ingredients:
                    # 진짜 화학성분명이 있으면 formula 소스 우선 + 성분명만으로 검색
                    payload = self.engine.search_bm25_ingredient_priority(
                        real_ingredients, k)
                else:
                    # '백탁'처럼 도메인/주제어만 매칭됐으면 기존 전역 검색 그대로
                    # (커머스 리뷰가 오히려 정답인 경우라 우선순위를 걸면 안 됨)
                    payload = self.engine.search_bm25(question, k)
            else:  # vector
                payload = self.engine.search_vector(question, k)

        evidence_text, citations = self._format_evidence(route, payload)
        has_evidence = bool(citations)

        prompt = f"질문: {question}\n\n근거 (route={route}):\n{evidence_text}\n"
        if route == "vector" and citations:
            prompt += f"\n{self._token_df_report(question)}\n"
        if not has_evidence:
            prompt += "\n주의: 위 근거가 비어 있습니다. 규칙 3에 따라 답할 수 없다고 답하세요."

        return GeneratedAnswer(question=question, route=route, prompt=prompt,
                               evidence_count=len(citations), citations=citations)

    def _search_temporal(self, decision: RouteDecision, k: int) -> dict:
        """시현님 모듈에 대응 없음 - 원래 로직 유지(날짜 메타데이터 필터).
        118MB 전체 DB 대신 필요한 컬럼만 뽑은 mfds_items.csv 를 쓴다."""
        import pandas as pd
        from datetime import timedelta

        df_all = pd.read_csv("mfds_items.csv", parse_dates=["report_date"])
        max_date = df_all["report_date"].max()
        cutoff = max_date - timedelta(days=decision.temporal_window_days)
        df = df_all[df_all["report_date"] >= cutoff].sort_values(
            "report_date", ascending=False)

        sub = self.router.classify(decision.residual_query)
        if sub.matched_ingredients or sub.matched_brands:
            terms = sub.matched_ingredients + sub.matched_brands
            mask = df["ITEM_NAME"].fillna("").apply(
                lambda t: any(term in t for term in terms if term not in ("선크림", "썬크림")))
            if mask.any():
                df = df[mask]

        return {
            "window": f"{cutoff.date()} ~ {max_date.date()} "
                     f"(최근 {decision.temporal_window_days}일)",
            "n_total_in_window": len(df),
            "results": df.head(k).to_dict("records"),
        }

    # -------------------------------------------------------------
    def generate(self, question: str, k: int = 8,
                force_multi_source: bool = False) -> GeneratedAnswer:
        result = self.build_prompt(question, k=k, force_multi_source=force_multi_source)

        if result.evidence_count == 0:
            result.answer = "제공된 데이터로는 답할 수 없습니다. 관련 근거를 찾지 못했습니다."
            result.executed = True
            result.note = "근거 0건 - LLM 호출 없이 규칙 3 자동 적용"
            return result

        if not self.api_key:
            result.note = ("ANTHROPIC_API_KEY 없음 - 실제 생성은 안 됨. "
                          "prompt 필드에 구성된 전체 프롬프트가 들어있음.")
            return result

        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-5", max_tokens=1024,
            system=SYSTEM_PROMPT, messages=[{"role": "user", "content": result.prompt}],
        )
        result.answer = msg.content[0].text
        result.executed = True
        return result


def demo():
    gen = AnswerGenerator()
    tests = [
        ("보고번호 2018008612 이거 언제 등록된 품목이야?", False),
        ("최근 나온 무기자차 선크림 뭐 있어?", False),
        ("백탁 관련해서 소비자들이 뭐라고 해?", True),
        ("크소나이드 함유 제품 있어?", False),  # df_gate 차단 케이스
        ("하얗게", False),  # 벡터가 이기는 자리 (짧은 별칭)
    ]
    for q, multi in tests:
        print("=" * 70)
        print("질문:", q)
        r = gen.generate(q, force_multi_source=multi)
        print("라우팅:", r.route, "| 근거 건수:", r.evidence_count)
        print("실행됨:", r.executed, "| 노트:", r.note)
        if r.answer:
            print("답변:", r.answer[:200])
        else:
            print("--- 구성된 프롬프트(앞부분) ---")
            print(r.prompt[:400])
        print()


if __name__ == "__main__":
    demo()
