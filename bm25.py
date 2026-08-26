#!/usr/bin/env python3
"""BM25 어휘 검색. RAG 검색 계층의 절반이고, 벡터 쪽의 기준선이다.

왜 어휘 검색이 필요한가. 벡터는 "뜻이 가까운 것"을 찾고 어휘는 "글자가 같은 것"을
찾는다. 우리 데이터에는 정확히 일치해야만 의미가 있는 말이 많다 —
`에칠헥실트리아존`, `SPF50+`, 브랜드명. 벡터에 넣으면 `에칠헥실메톡시신나메이트`
와 비슷하다고 나온다. 성분이 다른데 비슷하다고 하면 그건 틀린 답이다.

반대로 `하얗게 떠서 싫다` 는 어휘로는 못 찾는다. `백탁` 이라는 글자가 없다.
그래서 둘을 같이 쓰고 순위를 합친다(RRF). 이 파일은 어휘 쪽이다.

**벡터보다 이걸 먼저 만든 이유.** 임베딩 모델을 뭘로 정하든 이 기준선은 안 바뀐다.
기준선 없이 하이브리드를 만들면 "합친 게 더 낫다"를 확인할 방법이 없다.
어제 후향 검증에서 기저율을 같이 낸 것과 같은 이유다.

BM25 는 학습이 없다. 규칙 세 개다.
  1. 그 단어가 문서에 몇 번 나오나 (tf) — 많으면 관련 있다. 단 포화시킨다
  2. 그 단어가 전체 문서 몇 개에 나오나 (idf) — 흔하면 깎는다
  3. 문서가 얼마나 긴가 — 긴 문서는 아무 단어나 들어 있으니 보정한다

2번이 핵심이고, 어제 `unmatched_terms.py` 에서 lift 로 일반어를 깎은 것과 같은 발상이다.

토큰화는 언어로 갈린다. 한국어는 Kiwi 형태소(사용자 사전 필수), 그 외는 공백·소문자.
논문 데이터가 영어로 들어올 수 있어 자리를 미리 둔다. 토큰화가 소스마다 갈리면
점수를 비교할 수 없으므로 **언어 판정은 텍스트로만** 하고 소스로 하지 않는다.

입력은 `common/document.csv` 다. `source` 를 하드코딩하지 않으므로 커머스·논문·NAVER
가 같은 스키마로 변환되면 행이 늘어나는 것뿐이고 이 파일은 안 바뀐다.

사용법:
    python bm25.py --query "하얗게 떠서 싫다"
    python bm25.py --query "에칠헥실트리아존" --top 5
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from topics import TOPICS

csv.field_size_limit(10 ** 8)

K1 = 1.2      # tf 포화 계수. 표준값. 우리 데이터로 다시 뽑을 이유가 아직 없다
B = 0.75      # 길이 보정 강도. 표준값
# SPF50+ 를 한 토큰으로. 뒤에 아무것도 없는 `+` 도 살려야 한다 — `SPF50+` 가
# `spf` + `50` 으로 갈리면 차단지수 검색이 전부 어긋난다.
LATIN_RE = re.compile(r"[a-z0-9]+(?:[+\-][a-z0-9]+)*\+?")
HANGUL_RE = re.compile(r"[가-힣]")

# 내용어만 남긴다. 조사·어미·기호는 순위에 잡음만 넣는다.
# VA·VV 를 넣는 이유 — `끈적이다`·`시리다` 처럼 서술어가 주제인 경우가 많다.
# SL(외국어)·SN(숫자)은 **일부러 뺐다.** 라틴 토큰은 정규식이 담당한다. 둘 다 넣으면
# 같은 단어가 두 번 세어져 tf 가 부풀고, 한국어 문서에서만 영어 단어가 유리해진다.
#
# 태그는 `VA-I`·`VV-R` 처럼 불규칙 표시가 붙어 오므로 하이픈 앞만 본다. 이걸 놓쳐서
# `하얗게 떠서 싫다` 의 질의 토큰이 0개로 나왔다.
KIWI_TAGS = {"NNG", "NNP", "VA", "VV", "XR", "MAG"}

# 한 글자 명사는 잡음이다(거·것·수·때). 반면 **서술어 어간은 한 글자가 정상**이다 —
# `하얗`·`뜨`·`싫`·`시리`. 명사에만 길이 조건을 걸어야 한다.
NOUN_TAGS = {"NNG", "NNP"}

# 주제 별칭을 Kiwi 에 등록할 때 줄 가중치. 0 으로 두면 등록해도 기존 분석을 못 이겨
# `신제품` 이 신(XPN) + 제품(NNG) 으로 갈린다 — 조용히 무시된다.
USER_WORD_SCORE = 3.0

_kiwi = None
_topic_words: list[str] | None = None


def topic_words() -> list[str]:
    """주제 사전의 별칭 중 한 낱말인 것. Kiwi 에 통째로 등록해 쪼개지지 않게 한다.

    왜 필요한가. `topics.py` 는 **부분문자열**로 매칭하고 색인은 **형태소**로 나눈다.
    단위가 어긋나면 사전에 있는 말을 검색기가 못 찾는다. 실측으로 이랬다.

        신제품   -> ['제품']        `신` 이 접두사로 떨어진다
        화학적   -> ['화학']        `적` 이 접미사로 떨어진다
        차단지수 -> ['차단','지수']  둘로 갈린다
        제형     -> []             `제`(접두) + `형`(한 글자) 로 전멸한다

    `제형` 이 빈 결과인 것은 평가 문제가 아니라 검색 결함이다. 사용자가 `제형` 을
    치면 아무것도 안 나온다.

    **서술어 별칭은 등록하면 안 된다.** `하얗게`·`하얘` 를 명사로 넣으면 Kiwi 가
    둘을 각각 한 낱말로 주는데, 등록 전에는 둘 다 `하얗` 으로 통합됐다. 활용형을
    명사로 박으면 형태소 통합이 깨져서 `하얗게` 질의가 `하얘` 문서를 놓친다.

    그래서 **Kiwi 에게 먼저 물어보고 고른다** — 별칭 그대로 넣었을 때
      (1) 한 토큰으로 안 나오고
      (2) 서술어(VA·VV) 읽기가 없는
    것만 등록한다. 손으로 목록을 관리하면 주제 사전이 바뀔 때 어긋난다.

    공백이 든 별칭(`눈 시림`·`톤 업`)과 조사가 붙은 별칭(`땀에`)도 빠진다.
    낱말이 아니어서 Kiwi 에 넣을 수 없고, 부분문자열 사전에서만 의미가 있다.
    """
    global _topic_words
    if _topic_words is not None:
        return _topic_words

    from kiwipiepy import Kiwi
    bare = Kiwi()                       # 사용자 사전을 얹지 않은 판정용
    words = set()
    for entry in TOPICS:
        for alias in entry["ko"]:
            if " " in alias or len(alias) < 2 or alias.endswith("에"):
                continue
            tokens = bare.tokenize(alias)
            if len(tokens) == 1 and tokens[0].form == alias:
                continue                # 이미 한 낱말이다. 건드릴 이유가 없다
            if any(t.tag.split("-")[0] in {"VA", "VV"} for t in tokens):
                continue                # 활용형이다. 명사로 박으면 통합이 깨진다
            words.add(alias)
    _topic_words = sorted(words)
    return _topic_words


# 사전 두 개를 얹는다. 담론어(백탁·눈시림)와 성분명은 출처가 다르다.
# 성분 사전은 수호님이 성분표에서 뽑아 주신 것으로, 각주(`*`)와 안내문이 붙은
# 항목 21개를 걸러 1,877행을 쓴다. 이게 없으면 `에칠헥실트리아존` 같은 성분명이
# 형태소로 쪼개져 성분 검색이 조용히 안 된다.
DICTIONARIES = (Path("seeds/user_dictionary.tsv"),
                Path("seeds/ingredient_dictionary.tsv"))


def kiwi(dictionaries: tuple[Path, ...] = DICTIONARIES):
    """Kiwi 를 한 번만 만든다. 사전 없이 쓰면 백탁이 백+탁 으로 쪼개진다."""
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
        for dictionary in dictionaries:
            if dictionary.exists():
                _kiwi.load_user_dictionary(str(dictionary))
            else:
                print(f"[경고] 사전이 없다: {dictionary}")
        # 주제 별칭은 `topics.py` 가 정본이므로 TSV 에 복사하지 않고 여기서 넣는다.
        # 사전을 두 벌 두면 어느 쪽이 맞는지 알 수 없게 된다.
        for word in topic_words():
            # score 를 줘야 기존 분석을 이긴다. 0 으로 두면 등록해도 조용히
            # 무시되고 신제품이 신(XPN) + 제품(NNG) 으로 갈린다.
            _kiwi.add_user_word(word, "NNG", USER_WORD_SCORE)
    return _kiwi


def is_korean(text: str) -> bool:
    """한글이 5% 넘으면 한국어로 본다. 영어 논문에 한글 각주가 있어도 안 흔들린다."""
    if not text:
        return False
    return len(HANGUL_RE.findall(text)) / len(text) > 0.05


STOPWORDS_FILE = Path("seeds/stopwords_ko.txt")


def query_stopwords() -> frozenset[str]:
    """**질의에만** 적용하는 불용어. 색인에는 손대지 않는다.

    자연어로 물으면 필러가 끌고 간다 — `백탁 관련해서 소비자들이` 가
    `['관련','백탁','소비자']` 로 쪼개지고 *"후니는 우리 소비자편이야"* 가 3위에
    올라온다(수호님 발견). 색인에서 빼면 `소비자` 를 직접 찾는 질의를 못 하게 되니
    **질의 쪽만** 지운다.

    별칭 61개 중 이 목록과 겹치는 것은 **0개**라 기존 평가 숫자는 안 움직인다.
    질의가 전부 불용어면(예: `관련`) 지우지 않는다 — 빈 질의보다는 낫다.
    """
    if not STOPWORDS_FILE.exists():
        return frozenset()
    return frozenset(
        l.strip() for l in STOPWORDS_FILE.read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#"))


def tokenize_query(text: str) -> list[str]:
    """질의 토큰화. `tokenize` 에 불용어 제거만 얹는다."""
    toks = tokenize(text)
    kept = [t for t in toks if t not in query_stopwords()]
    return kept or toks          # 전부 불용어면 원본을 쓴다


def tokenize(text: str) -> list[str]:
    """언어로 갈린다. 두 갈래 모두 소문자 NFKC 를 거쳐 같은 표면형을 만든다."""
    text = unicodedata.normalize("NFKC", text or "")
    if not is_korean(text):
        # 영어·숫자 전용. 논문이 영어로 오면 이쪽을 탄다
        return LATIN_RE.findall(text.lower())
    out = []
    for token in kiwi().tokenize(text):
        tag = token.tag.split("-")[0]          # VA-I -> VA
        if tag not in KIWI_TAGS:
            continue
        if tag in NOUN_TAGS and len(token.form) < 2:
            continue
        out.append(token.form.lower())
    # 한국어 문서 안의 라틴 토큰(SPF50+, Tinosorb)은 정규식이 담당한다.
    # Kiwi 는 기호가 붙으면 쪼개므로 `SPF50+` 를 한 덩어리로 못 준다.
    out.extend(t for t in LATIN_RE.findall(text.lower()) if len(t) >= 2)
    return [t for token in out for t in expand(token)]


_expanded: dict[str, tuple[str, ...]] = {}
_expand_words: list[str] | None = None


def expand_words() -> list[str]:
    """부분문자열 확장에 쓸 별칭 전부. **등록 목록과 다른 집합이다.**

    등록(`topic_words`)은 "Kiwi 가 쪼개니까 붙여 달라"는 요청이고, 확장은
    "Kiwi 가 한 낱말로 잘 주는데 그 안에 별칭이 들어 있다"는 반대 상황이다.
    `끈적임` 이 그 예다 — Kiwi 가 온전히 주므로 등록 대상이 아니지만, 질의
    `끈적` 이 그 문서를 찾으려면 확장이 필요하다.
    """
    global _expand_words
    if _expand_words is None:
        _expand_words = sorted({a for e in TOPICS for a in e["ko"]
                                if " " not in a and len(a) >= 2})
    return _expand_words


def expand(token: str) -> tuple[str, ...]:
    """토큰이 주제 별칭을 품고 있으면 별칭도 같이 낸다.

    `topics.py` 는 부분문자열로 매칭하므로 **색인도 같은 단위를 만들어야** 한다.
    질의와 문서 양쪽에 똑같이 적용되므로 대칭이 깨지지 않는다. 고유 토큰 단위로
    기억해 두므로 별칭 검사를 12만 토큰에 대해 한 번만 한다.
    """
    hit = _expanded.get(token)
    if hit is None:
        extra = [w for w in expand_words() if w != token and w in token]
        hit = (token, *extra)
        _expanded[token] = hit
    return hit


class Index:
    """역색인 하나. 문서 수가 10만 규모라 메모리에 그냥 둔다."""

    def __init__(self, doc_ids: list[str], texts: list[str]):
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids 와 texts 길이가 다르다")
        self.doc_ids = doc_ids
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            self.lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))
        self.n = len(doc_ids)
        self.avg_len = (sum(self.lengths) / self.n) if self.n else 0.0
        self.position = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    def state(self) -> dict:
        """캐시에 담을 알맹이. 클래스가 아니라 dict 로 오간다."""
        return {"doc_ids": self.doc_ids, "postings": dict(self.postings),
                "lengths": self.lengths}

    @classmethod
    def from_state(cls, state: dict) -> "Index":
        index = cls.__new__(cls)
        index.doc_ids = state["doc_ids"]
        index.postings = state["postings"]
        index.lengths = state["lengths"]
        index.n = len(index.doc_ids)
        index.avg_len = (sum(index.lengths) / index.n) if index.n else 0.0
        index.position = {d: i for i, d in enumerate(index.doc_ids)}
        return index

    def idf(self, term: str) -> float:
        """Robertson–Sparck Jones. 절반 넘는 문서에 나오면 음수가 되므로 0 에서 막는다."""
        df = len(self.postings.get(term, ()))
        if df == 0:
            return 0.0
        return max(0.0, math.log((self.n - df + 0.5) / (df + 0.5) + 1.0))

    def search(self, query: str, k: int | None = 10,
               skip: set[str] | None = None) -> list[tuple[str, float]]:
        """상위 k 개 (doc_id, 점수). skip 은 후보에서 제외할 doc_id 집합이다.

        skip 이 필요한 이유는 평가다. 별칭 하나를 질의로 주고 **그 별칭이 글자로
        들어 있는 문서를 후보에서 빼면**, 글자가 겹치지 않는 같은 주제 문서를
        찾아낼 수 있는지 잴 수 있다. 벡터가 이겨야 하는 판이 정확히 이것이다.
        """
        banned = {self.position[d] for d in (skip or ()) if d in self.position}
        scores: dict[int, float] = defaultdict(float)
        # **질의는 `tokenize_query`.** 색인은 `tokenize` 그대로다 — 필러를 색인에서
        # 빼면 `소비자` 를 직접 찾는 질의를 못 하게 된다
        for term in set(tokenize_query(query)):
            weight = self.idf(term)
            if weight == 0.0:
                continue
            for i, tf in self.postings[term]:
                if i in banned:
                    continue
                norm = tf + K1 * (1 - B + B * self.lengths[i] / (self.avg_len or 1))
                scores[i] += weight * tf * (K1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self.doc_ids[kv[0]]))
        if k is not None:
            ranked = ranked[:k]
        return [(self.doc_ids[i], round(s, 4)) for i, s in ranked]


def load_documents(common: Path, sources: list[str] | None = None
                   ) -> tuple[list[str], list[str], dict[str, str]]:
    """(doc_ids, texts, doc_id -> source). quality_flags 가 붙은 행은 뺀다.

    `source` 로 필터할 수 있게만 두고 기본은 전부다. 커머스·논문이 붙으면
    같은 파일에 행이 늘어난다.
    """
    doc_ids, texts, origin = [], [], {}
    path = common / "document.csv"
    if not path.exists():
        # 청크만 색인하는 경우다. 경고만 하고 계속한다 — 조용히 빈 색인을
        # 만들면 "왜 아무것도 안 나오나" 를 찾는 데 시간을 쓴다
        print(f"[알림] {path} 가 없다. 청크 파일만 색인한다")
        return doc_ids, texts, origin
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["quality_flags"]:
                continue
            if sources and row["source"] not in sources:
                continue
            doc_ids.append(row["doc_id"])
            texts.append(row["text"])
            origin[row["doc_id"]] = row["source"]
    return doc_ids, texts, origin


def by_source(index: "Index", origin: dict[str, str], query: str,
              k: int = 5) -> dict[str, list[tuple[str, float]]]:
    """소스별 상위 k. **근거 도구는 이걸 써야 한다.**

    전역 상위 k 를 쓰면 다수 소스가 나머지를 덮는다. 실측으로 색인 269,851개 중
    유튜브 댓글이 247,086개(92%)고 평균 16토큰이라, BM25 의 길이 보정 때문에
    짧은 댓글이 상위를 독점한다. `판테놀 쓰는 선크림` 질의에서 소스별 첫 등장이
    이랬다.

        1위   youtube_comment
       37위   youtube_video
      132위   formula_full
      293위   mfds
      (ingredient · formula_summary 는 300위 안에 없음)

    이건 점수가 틀린 게 아니라 **묻는 방식이 틀린 것**이다. "무엇이 가장 관련
    있나" 가 아니라 "각 소스에서 무엇이 관련 있나" 를 물어야 근거가 모인다.
    소스 하나에서 근거를 다 뽑으면 교차 확인이 안 된다.

    **상위 N 을 잘라서 버킷에 담으면 안 된다.** 처음에 k*100 으로 잘랐더니
    `mfds`(첫 등장 293위)와 `ingredient`(300위 밖)가 여전히 안 나왔다. 점수가
    붙은 문서를 **전부** 받아서 소스별로 채운다 — 어차피 정렬 비용만 든다.
    """
    picked: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for doc_id, score in index.search(query, None):
        source = origin.get(doc_id, "?")
        if len(picked[source]) < k:
            picked[source].append((doc_id, score))
    return dict(picked)


def load_chunks(paths: list[Path]) -> tuple[list[str], list[str], dict[str, str]]:
    """청크 CSV 를 문서처럼 읽는다. 색인 단위는 `chunk_id` 다.

    왜 필요한가. `document.csv` 는 유튜브만 들어 있다. 성분·식약처는 `chunks.py`
    계약(5칸) 파일로만 존재하고 공통 스키마 변환기가 아직 없다. 그 변환기를
    기다리면 발표까지 검색이 유튜브만 찾는다.

    청크를 색인 단위로 쓰는 게 오히려 맞다 — 벡터 쪽도 청크 단위로 인코딩하므로
    두 검색기가 **같은 단위**를 본다. RRF 로 순위를 합칠 때 단위가 다르면 못 합친다.
    """
    doc_ids, texts, origin = [], [], {}
    for path in paths:
        if not path.exists():
            print(f"[경고] 청크 파일이 없다: {path}")
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                doc_ids.append(row["chunk_id"])
                texts.append(row["text"])
                origin[row["chunk_id"]] = row["source"]
    return doc_ids, texts, origin


def build(common: Path, sources: list[str] | None = None,
          cache: Path | None = Path(".cache/bm25"),
          chunks: list[Path] | None = None) -> tuple[Index, dict[str, str]]:
    """색인을 만들거나 캐시에서 읽는다.

    26만 문서를 Kiwi 로 훑는 데 4분 30초 걸린다. 평가는 색인 하나로 질의를 수십 개
    돌리는 일이므로 매번 다시 만들면 실험을 못 한다. 입력 파일과 토큰화 규칙의
    해시를 키로 두어, **규칙을 고치면 캐시가 자동으로 무효**가 되게 한다.
    """
    def gather() -> tuple[list[str], list[str], dict[str, str]]:
        doc_ids, texts, origin = load_documents(common, sources)
        if chunks:
            more_ids, more_texts, more_origin = load_chunks(chunks)
            doc_ids += more_ids
            texts += more_texts
            origin.update(more_origin)
        return doc_ids, texts, origin

    if cache is None:
        doc_ids, texts, origin = gather()
        return Index(doc_ids, texts), origin

    # **`document.csv` 가 없어도 돌아야 한다.** 청크만으로 색인하는 건 정당한
    # 사용이다 — 수호님처럼 커머스·성분만 쓰는 경우가 그렇다. 없으면 0 으로 둔다
    source_csv = common / "document.csv"
    src_stat = (f"{source_csv.stat().st_size}:{source_csv.stat().st_mtime_ns}"
                if source_csv.exists() else "no-document-csv")
    stamp = hashlib.sha256(
        f"{src_stat}"
        f":{sorted(sources or [])}"
        # 토큰화 규칙이 바뀌면 캐시를 버려야 한다. 안 그러면 옛 토큰으로 평가한다
        f":{sorted(KIWI_TAGS)}:{sorted(NOUN_TAGS)}:{LATIN_RE.pattern}:{K1}:{B}"
        f":{topic_words()}:{expand_words()}:{USER_WORD_SCORE}"
        # 사전이 바뀌면 토큰이 바뀐다. 해시를 키에 넣지 않으면 옛 색인을 계속 쓴다
        f":{[hashlib.sha256(d.read_bytes()).hexdigest()[:12] for d in DICTIONARIES if d.exists()]}"
        # 청크 파일도 색인 내용이다. 빠뜨리면 파일을 바꿔도 옛 색인을 쓴다
        f":{[(str(c), c.stat().st_size, c.stat().st_mtime_ns) for c in (chunks or []) if c.exists()]}"
        .encode()).hexdigest()[:16]
    path = cache / f"index-{stamp}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            state = pickle.load(handle)
        return Index.from_state(state), state["origin"]

    doc_ids, texts, origin = gather()
    index = Index(doc_ids, texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        # 클래스 인스턴스를 그대로 절이면 안 된다. bm25.py 를 직접 실행해 만든
        # 캐시는 `__main__.Index` 로 기록되고, import 해서 읽을 때 못 찾는다.
        # 평소 dict 만 담는다.
        pickle.dump({**index.state(), "origin": origin}, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return index, origin


def demo() -> None:
    # 언어 판정
    assert is_korean("백탁이 심해요") and not is_korean("photostability of avobenzone")
    assert not is_korean("")
    assert is_korean("SPF 측정 in vitro 결과")          # 한글 섞이면 한국어
    # 라틴 토큰은 기호를 살린다 — SPF50+ 가 SPF / 50 으로 갈리면 검색이 안 된다
    assert "spf50+" in tokenize("SPF50+ 제품")
    assert "tinosorb" in tokenize("Tinosorb S 함유")
    # 같은 라틴 단어가 두 번 세어지면 tf 가 부푼다 (Kiwi SL + 정규식 중복)
    assert tokenize("Tinosorb 함유").count("tinosorb") == 1
    # 한 글자 라틴은 버린다 — `S`, `A` 가 상위에 올라온다
    assert "s" not in tokenize("Tinosorb S 함유")
    # 사용자 사전 확인. 이게 깨지면 검색 전체가 조용히 망가진다
    assert "백탁" in tokenize("백탁 없이 촉촉해요"), tokenize("백탁 없이 촉촉해요")
    # 성분명은 통째로 한 토큰이어야 한다. 쪼개지면 성분 검색이 안 된다
    for name in ("에칠헥실트리아존", "나이아신아마이드", "판테놀", "징크옥사이드"):
        assert name in tokenize(f"{name} 함유 제품"), (name, tokenize(name))
    # 조사·어미는 빠져야 한다
    assert "이" not in tokenize("백탁이 심해요")
    # 주제 별칭은 통째로 한 토큰이어야 한다. 쪼개지면 사전에 있는 말을 못 찾는다
    assert tokenize("제형이 좋다") and "제형" in tokenize("제형이 좋다")
    assert "신제품" in tokenize("신제품 출시")
    assert "차단지수" in tokenize("차단지수 높은 제품")
    assert "피부톤" in tokenize("피부톤 보정")
    # 낱말이 아닌 별칭은 등록하지 않는다
    assert "땀에" not in topic_words() and "눈 시림" not in topic_words()
    # 서술어 어간은 한 글자여도 살려야 한다. 이걸 버려서 질의 토큰이 0개가 됐다
    assert tokenize("하얗게 떠서 싫다"), "형용사·동사만 있는 질의가 비면 검색이 안 된다"
    assert "하얗" in tokenize("하얗게 떠서 싫다")
    assert "시리" in tokenize("눈이 시려요")
    # 확장 — 끈적임 은 Kiwi 가 한 낱말로 주므로 별칭 끈적 을 같이 내야 한다
    assert "끈적" in tokenize("끈적임 심함") and "끈적임" in tokenize("끈적임 심함")
    assert expand("끈적임") == ("끈적임", "끈적")
    assert expand("백탁") == ("백탁",)          # 자기 자신은 중복으로 넣지 않는다
    assert expand("아무말") == ("아무말",)
    # 등록 목록과 확장 목록은 다른 집합이다. 섞으면 둘 중 하나가 망가진다
    assert "끈적" in expand_words() and "끈적" not in topic_words()
    assert "제형" in topic_words() and "제형" in expand_words()
    # 활용형 별칭은 등록하지 않는다 — 등록하면 하얗게/하얘 통합이 깨진다
    assert "하얗게" not in topic_words() and "하얘" not in topic_words()
    assert tokenize("하얗게") == tokenize("하얘")
    # 한 글자 명사는 여전히 버린다
    assert "것" not in tokenize("이런 것 좋아요")

    index = Index(
        ["a", "b", "c"],
        ["백탁 없이 촉촉하다", "끈적임이 심하다 유분감", "백탁 백탁 백탁 하얗게 뜬다"],
    )
    top = index.search("백탁", k=3)
    assert top[0][0] == "c", top          # 세 번 나온 문서가 먼저
    assert [d for d, _ in top] == ["c", "a"], top
    # 흔한 말은 idf 가 0 이라 점수를 못 만든다 — 세 문서 중 하나에만 있어야 걸린다
    assert index.search("없다", k=3) == []
    # skip 은 후보에서 빼는 것이다
    assert index.search("백탁", k=3, skip={"c"})[0][0] == "a"
    assert index.search("백탁", k=3, skip={"a", "c"}) == []
    # 없는 말은 빈 결과. 예외를 던지면 평가 루프가 멈춘다
    assert index.search("존재하지않는성분명", k=3) == []
    # k=None 은 점수 붙은 것 전부. 소스별로 뽑을 때 잘라 내면 소수 소스를 놓친다
    assert len(index.search("백탁", None)) == 2 and len(index.search("백탁", 1)) == 1
    # 소스별 뽑기 — 다수 소스가 나머지를 덮지 않아야 한다
    many = Index([f"c{i}" for i in range(30)] + ["ing1", "ing2"],
                 ["백탁 심하다"] * 30 + ["백탁 성분 정보", "백탁 성분 자료"])
    origin = {**{f"c{i}": "comment" for i in range(30)},
              "ing1": "ingredient", "ing2": "ingredient"}
    got = by_source(many, origin, "백탁", k=2)
    assert set(got) == {"comment", "ingredient"}, got
    assert len(got["comment"]) == 2 and len(got["ingredient"]) == 2, got

    # 캐시 왕복. dict 로 오가지 않으면 __main__.Index 로 절여져 import 시 못 읽는다
    same = Index.from_state(index.state())
    assert same.search("백탁", k=3) == index.search("백탁", k=3)
    assert abs(same.avg_len - index.avg_len) < 1e-9 and same.n == index.n
    print("demo ok")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--common", type=Path, default=Path("common"))
    p.add_argument("--query")
    p.add_argument("--source", action="append",
                   help="youtube_video / youtube_comment 등. 여러 번 쓸 수 있다")
    p.add_argument("--top", type=int, default=10)
    # **`action="append"` 에 default 를 주면 안 된다.** argparse 는 기본값을 지우지
    # 않고 거기에 덧붙인다 — `--chunks A` 가 [기본값, A] 가 되어 성분·식약처가
    # 조용히 같이 색인된다. 소스를 갈라 색인하려는 사람이 절대 못 가른다.
    # None 으로 두고 아래에서 채운다
    p.add_argument("--chunks", action="append", type=Path,
                   help="청크 CSV. 성분·식약처는 공통 스키마 변환기가 없어 이쪽으로 넣는다. "
                        "안 주면 reports/chunks_ingredient_mfds.csv 하나를 쓴다")
    p.add_argument("--cache", default=".cache/bm25")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--per-source", action="store_true",
                   help="소스별 상위 k. 근거 도구는 이쪽이다")
    p.add_argument("--demo", action="store_true")
    a = p.parse_args()
    if a.demo:
        demo()
        return 0
    if not a.query:
        p.error("--query 를 주거나 --demo 를 쓴다")

    chunks = a.chunks if a.chunks else [Path("reports/chunks_ingredient_mfds.csv")]
    index, origin = build(a.common, a.source,
                          None if a.no_cache else Path(a.cache), chunks)
    print(f"색인 {index.n:,}개 문서 · 고유 토큰 {len(index.postings):,} · "
          f"평균 길이 {index.avg_len:.1f}")
    toks = sorted(set(tokenize_query(a.query)))
    dropped = sorted(set(tokenize(a.query)) - set(toks))
    print(f"질의 토큰: {toks}" + (f"  (불용어로 뺀 것: {dropped})" if dropped else ""))
    print()

    if a.per_source:
        ids, bodies, _o = load_documents(a.common, a.source)
        if a.chunks:
            more_ids, more_bodies, _ = load_chunks(a.chunks)
            ids += more_ids
            bodies += more_bodies
        body = dict(zip(ids, bodies))
        found = by_source(index, origin, a.query, a.top)
        for source in sorted(found, key=lambda s: -found[s][0][1]):
            print(f"[{source}]")
            for doc_id, score in found[source]:
                snippet = body[doc_id][:104].replace("\n", " ")
                print(f"  {score:>7.2f}  {snippet}")
        missing = sorted(set(origin.values()) - set(found))
        if missing:
            print(f"(걸린 문서 없음: {', '.join(missing)})")
        return 0

    ids, bodies, _origin = load_documents(a.common, a.source)
    if a.chunks:
        more_ids, more_bodies, _ = load_chunks(a.chunks)
        ids += more_ids
        bodies += more_bodies
    texts = dict(zip(ids, bodies))
    for rank, (doc_id, score) in enumerate(index.search(a.query, a.top), 1):
        snippet = texts[doc_id][:110].replace("\n", " ")
        print(f"{rank:>2}. {score:>8.3f}  {origin[doc_id]:<16}{snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
