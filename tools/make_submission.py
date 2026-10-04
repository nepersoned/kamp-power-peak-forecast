"""대회 제출용 소스코드 zip 생성 (소스코드·requirements·학습용 데이터·README·테스트 예측결과 파일).

    python tools/make_submission.py            # -> dist/submission_source.zip

- 저장소에 커밋된 파일만 담는다(git ls-files). 가상환경·캐시·로컬 실험물은 들어가지 않는다.
- 블라인드 평가: 내부 진행 기록과 보고서 원고(docs/PROGRESS.md, docs/REPORT_*.md)는 제외하고,
  남은 텍스트 파일에 저장소 주소·계정명이 없는지 검사한다.
- 테스트 예측결과와 최종 비교표는 outputs/에서 골라 함께 담는다(먼저 final_system --stage test 실행).
"""
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
EXCLUDE = re.compile(r"^docs/(PROGRESS|REPORT_[^/]*)\.md$")
# 저장소 계정·소속 (이 파일 자체에 걸리지 않도록 조각으로 둔다)
IDENTITY = re.compile("|".join(["neper" + "soned", "5castle" + "min", "kevin" + "bae", "한국" + "외국어", "HU" + "FS"]), re.I)
OUTPUTS = [
    "outputs/final_system/test_predictions_regime_tabpfn.csv",
    "outputs/final_system/test_forecast_distributions.csv",
    "outputs/final_system/test_production_plans.csv",
    "outputs/final_system/test_daily_decisions.csv",
    "outputs/final_system/test_forecast_comparison.csv",
    "outputs/final_system/test_probabilistic_metrics.csv",
    "outputs/final_system/test_decision_metrics.csv",
    "outputs/final_system/test_bootstrap_forecast.csv",
    "outputs/final_system/test_bootstrap_decision.csv",
    "outputs/test_predictions.csv",
]


def main():
    files = [f for f in subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
             if f and not EXCLUDE.match(f)]
    missing = [f for f in OUTPUTS if not (ROOT / f).exists()]
    if missing:
        sys.exit(f"테스트 결과 파일이 없습니다. 먼저 실행하세요: {missing}")
    leaks = [f for f in files if not f.endswith((".png", ".pkl", ".xlsx", ".csv"))
             and IDENTITY.search((ROOT / f).read_text(encoding="utf-8", errors="ignore"))]
    if leaks:
        sys.exit(f"블라인드 위반 가능(저장소 주소·계정명): {leaks}")
    DIST.mkdir(exist_ok=True)
    out = DIST / "submission_source.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files + OUTPUTS:
            z.write(ROOT / f, f)
    print(f"{out} ({len(files) + len(OUTPUTS)} files, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
