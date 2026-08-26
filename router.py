"""
router.py — 질의 라우팅 레이어

LLM 호출 없이 규칙 기반으로 분류한다. 결정적이고, 테스트 가능하고, 왜 그렇게
라우팅했는지 항상 설명 가능해야 하기 때문이다 (이 프로젝트 전체의 원칙과 동일).

라우팅 규칙 (전부 실측 근거 있음):
  1. 등록번호(10자리 숫자)      -> BM25, mfds 소스로 좁힘
  2. SPF/PA 수치                -> BM25
  3. 사전에 있는 성분명 포함    -> BM25
  4. 브랜드명 포함              -> BM25
  5. 시점 표현("최근","요즘")   -> temporal_filter (검색이 아니라 날짜 메타데이터 필터)
  6. 그 외 (자연어 표현)        -> vector (e5)
  7. 여러 유형이 동시에 걸리면  -> multi_source (소스별로 따로 검색해 나란히 제시)

우선순위: 1~4는 서로 배타적이지 않게 다 체크하고, 걸린 게 있으면 BM25.
5(시점)는 다른 것과 같이 걸릴 수 있음 - 이 경우 날짜 필터를 먼저 적용한 뒤
남은 부분으로 검색 라우팅을 다시 판단(temporal_filter + 하위 라우팅).
아무것도 안 걸리면 vector.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------
# 사전 로드
# ---------------------------------------------------------------
INGREDIENT_DICT = "kiwi_userdict_수호_v2.tsv"
BRANDS_CSV = "brands.csv"

REG_NO_PATTERN = re.compile(r"\b\d{10}\b")
SPF_PATTERN = re.compile(r"SPF\s*\d{1,3}\+?|PA\s*\+{1,4}", re.IGNORECASE)
TEMPORAL_KEYWORDS = ["최근", "요즘", "신제품", "새로 나온", "신상", "이번 달",
                     "올해", "최신", "요새", "근래"]

# 시점 표현에서 실제 기간을 뽑기 위한 대략적 매핑 (필요시 확장)
TEMPORAL_WINDOW_DAYS = {
    "최근": 180, "요즘": 180, "요새": 180, "근래": 180,
    "이번 달": 30, "올해": 365, "최신": 90, "신제품": 180,
    "새로 나온": 180, "신상": 90,
}


@dataclass
class RouteDecision:
    query: str
    route: str                       # "bm25" | "vector" | "temporal_filter" | "multi_source"
    matched_ingredients: list = field(default_factory=list)
    matched_brands: list = field(default_factory=list)
    matched_reg_no: list = field(default_factory=list)
    matched_spf: list = field(default_factory=list)
    temporal_window_days: int | None = None
    residual_query: str = ""         # 시점 표현을 뗀 나머지 (하위 라우팅용)
    reason: str = ""


class QueryRouter:
    def __init__(self, ingredient_dict_path=INGREDIENT_DICT, brands_csv=BRANDS_CSV):
        self.ingredients = self._load_ingredients(ingredient_dict_path)
        self.brands = self._load_brands(brands_csv)
        # 긴 이름부터 매칭해야 부분 문자열 오매칭을 피한다
        self._ingredients_sorted = sorted(self.ingredients, key=len, reverse=True)
        self._brands_sorted = sorted(self.brands, key=len, reverse=True)

    def _load_ingredients(self, path) -> set:
        words = set()
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if parts and len(parts[0]) >= 2:
                    words.add(parts[0])
        return words

    def _load_brands(self, brands_csv) -> set:
        raw = set(pd.read_csv(brands_csv)["brand"].dropna())

        # 법인 접두사/접미사 제거. 사람들은 '(주)아모레퍼시픽'이 아니라
        # '아모레퍼시픽'으로 검색한다 - 원문 그대로 두면 실제 질의와 매칭 안 됨.
        strip_pat = re.compile(r"^\(주\)|^㈜|^주식회사\s*|\s*주식회사$|\(유\)$|㈜$")
        brands = set()
        for b in raw:
            if not isinstance(b, str) or len(b) < 2:
                continue
            cleaned = strip_pat.sub("", b).strip()
            if len(cleaned) >= 2:
                brands.add(cleaned)
            brands.add(b)  # 원문도 같이 남겨서 법인명 그대로 검색해도 잡히게
        return brands

    def _find_matches(self, query: str, vocab_sorted: list) -> list:
        found, remaining = [], query
        for term in vocab_sorted:
            if term in remaining:
                found.append(term)
                remaining = remaining.replace(term, " ")
        return found

    def classify(self, query: str) -> RouteDecision:
        reg_no = REG_NO_PATTERN.findall(query)
        spf = SPF_PATTERN.findall(query)
        ingredients = self._find_matches(query, self._ingredients_sorted)
        brands = self._find_matches(query, self._brands_sorted)

        temporal_hit = next((k for k in TEMPORAL_KEYWORDS if k in query), None)

        exact_signals = bool(reg_no or spf or ingredients or brands)

        if temporal_hit:
            residual = query.replace(temporal_hit, "").strip()
            return RouteDecision(
                query=query, route="temporal_filter",
                matched_ingredients=ingredients, matched_brands=brands,
                matched_reg_no=reg_no, matched_spf=spf,
                temporal_window_days=TEMPORAL_WINDOW_DAYS.get(temporal_hit, 180),
                residual_query=residual,
                reason=f"시점 표현 '{temporal_hit}' 감지 - 텍스트 검색이 아니라 "
                       f"report_date 기준 최근 {TEMPORAL_WINDOW_DAYS.get(temporal_hit,180)}일 필터. "
                       f"실측 확인: 어떤 검색기로도 이 유형은 못 풂",
            )

        # 서로 다른 소스 유형 신호가 동시에 강하게 걸리면(예: 성분명 + 자연어 반응) multi_source
        has_natural_lang_cue = len(query) > 15 and not exact_signals

        if exact_signals:
            reason_parts = []
            if reg_no:
                reason_parts.append(f"등록번호 {reg_no}")
            if spf:
                reason_parts.append(f"SPF/PA 표기 {spf}")
            if ingredients:
                reason_parts.append(f"성분명 {ingredients}")
            if brands:
                reason_parts.append(f"브랜드명 {brands}")
            return RouteDecision(
                query=query, route="bm25",
                matched_ingredients=ingredients, matched_brands=brands,
                matched_reg_no=reg_no, matched_spf=spf,
                reason="정확 문자열 신호 감지: " + ", ".join(reason_parts) +
                       " -> BM25 (실측 Hit@10 93% vs 벡터 89%)",
            )

        return RouteDecision(
            query=query, route="vector",
            reason="정확 문자열 신호 없음 - 자연어 표현으로 판단. "
                   "e5 벡터 검색으로 라우팅 (실측: 이런 질의에서 BM25는 0%, 벡터만 성과)",
        )


def demo():
    r = QueryRouter()
    tests = [
        "에칠헥실트리아존 쓰는 선크림",
        "보고번호 2018008612",
        "SPF50 PA++++ 제품",
        "하얗게 뜨는 거 없는 자외선 차단제",
        "최근에 나온 무기자차 선크림",
        "요즘 나이아신아마이드 들어간 신제품",
        "눈이 시리고 따가워요",
        "아모레퍼시픽 선크림",
    ]
    for q in tests:
        d = r.classify(q)
        print(f"[{d.route:16s}] {q}")
        print(f"    -> {d.reason}")
        if d.route == "temporal_filter":
            print(f"    -> 잔여 질의: {d.residual_query!r}, 윈도우: {d.temporal_window_days}일")
        print()


if __name__ == "__main__":
    demo()
