#!/usr/bin/env python3
"""선크림 주제 사전 v0.1 — 팀 공용 단일 소스.

이 파일이 사전의 정본이고, `--export`로 팀 공유용 CSV를 만든다.
소스별 담당자는 `match_topics(text)`를 import 해서 쓴다. 각자 사전을 만들면
`무기자차`가 사람마다 다르게 잡혀 소스 간 비교가 무의미해진다.

실측 근거 (518편 제목+설명+태그 / 댓글 1,907건, run_20260818T015739Z):
- 별칭 18개는 코퍼스 등장 0건이라 뺐다. 추측으로 넣으면 안 잡히는 단어가 섞인다.
- `산화아연`·`이산화티타늄`은 유튜브에서 0건인데 식약처는 이 이름을 쓴다.
  그래서 `mfds_inci_terms`를 따로 둔다. 이 열이 없으면 유튜브와 식약처가 안 붙는다.
- 2026-08-21 정정. 실제 전성분표(올리브영 선크림 368개, 30,097행)에서 확인해 보니
  표기가 `산화아연`이 아니라 **`징크옥사이드`(176개 제품)**, `이산화티타늄`이 아니라
  **`티타늄디옥사이드`(302개)**였다. 유기자차도 `아보벤존`이 아니라
  `부틸메톡시디벤조일메탄`(33개)이고, 우리가 적어 둔 `에틸헥실...`은 식약처 표기로
  `에칠헥실...`이다(`틸` 대 `칠`). 이 표기들로는 0건이 나왔다. 실측 표기를 앞에 두고
  기존 표기는 뒤에 남겨 둔다 — 다른 소스가 그 이름을 쓸 수 있다.
- 전성분표의 성분명에는 공백이 끼어드는 파싱 오류가 있다(`에칠헥실트 리아존`).
  성분으로 매칭할 때는 공백을 제거하고 비교한다. 성분 고유값이 2,182에서
  1,972로 줄고 UV 차단 성분은 19종에서 9종으로 합쳐진다.
- 영문 짧은 토큰은 경계 매칭이 필수다. `PA` 부분문자열은 `coupang`(제휴링크)에
  걸려 오탐 16%(188 -> 158편)가 났다.
- `선크림`(481/518=93%)·`추천`(396/518=76%)은 너무 흔해 판별력이 없다.
  트렌드 판정에서 제외하고 필터·장르 표시로만 쓴다.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable, Sequence

# topic, topic_type, korean terms(부분문자열), latin terms(경계 매칭),
# 식약처·INCI 표기, 트렌드 판정 사용 여부, 비고
TOPICS: list[dict] = [
    {
        "topic": "백탁", "topic_type": "attribute",
        "ko": ["백탁", "하얗게", "하얘"], "latin": [],
        "mfds_inci": [], "trend_use": True,
        "note": "백태·창백·유령·회끄무레는 실측 0건이라 제외",
    },
    {
        "topic": "자극_눈시림", "topic_type": "attribute",
        "ko": ["눈시림", "눈 시림", "눈따가", "자극", "따갑", "붉어", "뒤집", "예민", "트러블", "알러지"],
        "latin": [], "mfds_inci": [], "trend_use": True,
        "note": "눈시려·눈아파·눈매움 0건 제외. '눈물'은 감동 댓글과 섞여 제외",
    },
    {
        "topic": "발림성", "topic_type": "attribute",
        "ko": ["발림성", "발림", "제형", "텍스처"], "latin": [],
        "mfds_inci": [], "trend_use": True,
        "note": "'발림'은 '발림성'보다 17편 더 잡는다(발림이 좋다 등). 둘 다 유지",
    },
    {
        "topic": "촉촉함_건조함", "topic_type": "attribute",
        "ko": ["촉촉", "수분", "보습", "건조"], "latin": [],
        "mfds_inci": [], "trend_use": True,
        "note": "'건조'는 댓글(89)이 영상(38)보다 많다. 불만은 댓글에 쌓인다",
    },
    {
        "topic": "끈적임_유분감", "topic_type": "attribute",
        "ko": ["끈적", "유분", "번들", "기름"], "latin": [],
        "mfds_inci": [], "trend_use": True, "note": "미끌 0건 제외",
    },
    {
        "topic": "밀림_들뜸", "topic_type": "attribute",
        "ko": ["밀림", "밀려", "들뜸", "뭉침"], "latin": [],
        "mfds_inci": [], "trend_use": True, "note": "'화장 궁합' 0건 제외",
    },
    {
        "topic": "지속력_워터프루프", "topic_type": "attribute",
        "ko": ["지속력", "워터프루프", "무너짐", "재도포", "땀에", "방수"], "latin": [],
        "mfds_inci": [], "trend_use": True,
        "note": "'지속'만으로는 '지속적으로' 등 일반어에 걸려 제외",
    },
    {
        "topic": "톤업_메이크업베이스", "topic_type": "attribute",
        "ko": ["톤업", "톤 업", "메이크업베이스", "피부톤", "화이트닝"], "latin": [],
        "mfds_inci": [], "trend_use": True,
        "note": "'베이스'(83)는 단독으로 화장품 일반어라 제외",
    },
    {
        "topic": "무기자차", "topic_type": "formula",
        "ko": ["무기자차", "징크", "티타늄", "미네랄"], "latin": ["ZnO", "TiO2"],
        "mfds_inci": ["징크옥사이드", "티타늄디옥사이드",
                      "산화아연", "이산화티타늄", "Zinc Oxide", "Titanium Dioxide"],
        "trend_use": True,
        "note": "유튜브는 '무기자차', 식약처는 '산화아연'. 표기 겹침 0건 -> 매핑 필수",
    },
    {
        "topic": "유기자차", "topic_type": "formula",
        "ko": ["유기자차", "아보벤존", "옥토크릴렌", "화학적"], "latin": [],
        "mfds_inci": ["에칠헥실트리아존", "비스-에칠헥실옥시페놀메톡시페닐트리아진",
                      "메칠렌비스-벤조트리아졸릴테트라메칠부틸페놀", "에칠헥실살리실레이트",
                      "에칠헥실메톡시신나메이트", "부틸메톡시디벤조일메탄", "옥토크릴렌",
                      "디에칠아미노하이드록시벤조일헥실벤조에이트", "드로메트리졸트리실록산",
                      "테레프탈릴리덴디캠퍼설포닉애씨드", "페닐벤즈이미다졸설포닉애씨드",
                      "벤조페논-3", "4-메칠벤질리덴캠퍼",
                      "아보벤존", "에틸헥실메톡시신나메이트", "Avobenzone", "Octocrylene"],
        "trend_use": True, "note": "케미컬·유비놀 0건 제외",
    },
    {
        "topic": "혼합자차", "topic_type": "formula",
        "ko": ["혼합자차"], "latin": [],
        "mfds_inci": [], "trend_use": True,
        "note": "19편뿐. 분기당 표본 부족 가능성 높음 -> 판정 시 확인",
    },
    {
        "topic": "SPF_PA", "topic_type": "spec",
        "ko": ["차단지수"], "latin": ["SPF", "PA", "UVA", "UVB"],
        "mfds_inci": [], "trend_use": True,
        "note": "경계 매칭 필수. 부분문자열은 coupang에 걸려 오탐 16%",
    },
    {
        "topic": "성분_신제품", "topic_type": "event",
        "ko": ["성분", "신제품", "출시", "리뉴얼", "신상"], "latin": [],
        "mfds_inci": [], "trend_use": True,
        "note": "'신제품'만은 3편뿐. 전용 검색어 추가 대상(UnitA 4.1)",
    },
    {
        "topic": "추천_재구매", "topic_type": "genre",
        "ko": ["재구매", "품절", "인생템", "추천"], "latin": [],
        "mfds_inci": [], "trend_use": False,
        "note": "'추천' 396/518=76%. 장르 표시일 뿐 주제가 아니다. 트렌드 판정 제외",
    },
    {
        "topic": "선크림", "topic_type": "product_category",
        "ko": ["선크림", "썬크림", "자외선차단제", "선블록", "선스틱", "선쿠션", "선세럼", "선젤"],
        "latin": [], "mfds_inci": ["자외선차단제"], "trend_use": False,
        "note": "481/518=93%. 판별력 없음. 다른 소스에서 문서 필터로만 사용",
    },
]


def _latin_pattern(terms: Sequence[str]) -> re.Pattern[str] | None:
    """영문 토큰은 앞뒤가 영문자가 아닐 때만 매칭한다 (coupang -> PA 오탐 차단)."""
    if not terms:
        return None
    alts = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z]){alts}(?![A-Za-z])", re.IGNORECASE)


_LATIN = {t["topic"]: _latin_pattern(t["latin"]) for t in TOPICS}


def match_topics(text: str, *, include_excluded: bool = False) -> list[str]:
    """텍스트에 등장하는 주제 목록. 한 문서가 여러 주제에 걸릴 수 있다."""
    if not text:
        return []
    lowered = text.lower()
    hits = []
    for entry in TOPICS:
        if not entry["trend_use"] and not include_excluded:
            continue
        if any(term.lower() in lowered for term in entry["ko"]):
            hits.append(entry["topic"])
            continue
        pattern = _LATIN[entry["topic"]]
        if pattern and pattern.search(text):
            hits.append(entry["topic"])
    return hits


def export_csv(path: Path) -> int:
    rows = [
        {
            "topic": t["topic"],
            "topic_type": t["topic_type"],
            "youtube_terms": " | ".join(t["ko"] + t["latin"]),
            "match_mode": "boundary" if t["latin"] else "substring",
            "mfds_inci_terms": " | ".join(t["mfds_inci"]),
            "trend_use": "Y" if t["trend_use"] else "N",
            "note": t["note"],
        }
        for t in TOPICS
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def demo() -> None:
    """실데이터에서 뽑은 문장으로 사전이 도는지 확인한다."""
    assert "백탁" in match_topics("백탁없이 촉촉하게 발려요")
    assert "촉촉함_건조함" in match_topics("백탁없이 촉촉하게 발려요")
    assert "SPF_PA" in match_topics("SPF50+ PA++++ 제품입니다")
    # coupang 오탐 차단 — 이게 깨지면 SPF_PA가 제휴링크 영상 30편을 잘못 먹는다
    assert "SPF_PA" not in match_topics("구매링크 https://link.coupang.com/abc")
    assert "무기자차" in match_topics("징크 베이스 무기자차 제품")
    # 식약처 표기는 유튜브 용어와 겹치지 않으므로 별도 열로만 존재한다
    assert "무기자차" not in match_topics("산화아연 20% 함유")
    # 장르어는 기본 제외, 필요 시에만 포함
    assert "추천_재구매" not in match_topics("재구매 의사 있어요")
    assert "추천_재구매" in match_topics("재구매 의사 있어요", include_excluded=True)
    assert match_topics("") == []
    print("[demo] 통과")


def main() -> int:
    parser = argparse.ArgumentParser(description="선크림 주제 사전 v0.1")
    parser.add_argument("--export", default="seeds/topics_v0.1.csv", help="CSV 내보낼 경로")
    parser.add_argument("--demo", action="store_true", help="자체 점검만 실행")
    args = parser.parse_args()
    if args.demo:
        demo()
        return 0
    count = export_csv(Path(args.export))
    trend = sum(1 for t in TOPICS if t["trend_use"])
    print(f"[export] {args.export} — 주제 {count}개 (트렌드 판정용 {trend}개, 제외 {count - trend}개)")
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
