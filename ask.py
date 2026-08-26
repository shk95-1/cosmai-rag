"""
ask.py — 대화형 질의 (하드코딩된 테스트 케이스 없이 직접 질문)

generate.py 의 demo() 는 정해진 5개 질문만 돕니다. 이건 그 대신 터미널에서
원하는 질문을 자유롭게 넣을 수 있게 만든 버전입니다.

사용법:
    $env:ANTHROPIC_API_KEY = "..."   (PowerShell, 이미 하셨으면 생략)
    python ask.py

    질문: 백탁 없는 무기자차 선크림 있어?
    (답변 출력)

    질문: exit          <- 종료
"""
import sys

from generate import AnswerGenerator


def main():
    print("초기화 중 (색인 로딩·벡터 로딩)...")
    gen = AnswerGenerator()
    print("준비 완료. 질문을 입력하세요 ('exit' 또는 Ctrl+C로 종료)\n")

    while True:
        try:
            q = input("질문: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit", "종료"):
            print("종료합니다.")
            break

        # '멀티소스 강제'가 필요하면 질문 앞에 '!multi ' 를 붙이는 걸로 처리
        force_multi = False
        if q.startswith("!multi "):
            force_multi = True
            q = q[len("!multi "):].strip()

        try:
            r = gen.generate(q, force_multi_source=force_multi)
        except Exception as e:
            print(f"[에러] {e}\n")
            continue

        print(f"\n[라우팅: {r.route} | 근거 {r.evidence_count}건]")
        print(r.answer)
        if r.citations:
            print(f"\n(인용: {', '.join(r.citations[:5])}"
                 f"{' ...' if len(r.citations) > 5 else ''})")
        print()


if __name__ == "__main__":
    sys.exit(main())
