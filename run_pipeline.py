"""전처리부터 학습·추론·결과 생성까지 한 번에 실행하는 단일 진입점.

    python run_pipeline.py            # 전체(약 45분, CPU)
    python run_pipeline.py --quick    # 최종 시스템과 보고서 표만(약 30분)

순서
  1) pytest                       누수 감사·모델·시스템 테스트
  2) run_all.py                   비교 모델·오류분석·해석·그림 (--quick이면 생략)
  3) experiments.final_system     prepare → audit → test (동결 설정, 테스트 1회)
  4) forecast_model_search        검증 fold 예측(base, tabpfn) — 보고서 표의 검증 열
  5) src.final_analysis           tables · fnfp · peak (보고서 수치)
  6) src.report_figures           보고서 그림
이미 테스트를 실행한 출력 폴더가 있으면 3)의 test 단계는 재실행을 거부한다(테스트 1회 원칙).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def step(*args):
    print("\n==>", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def main():
    quick = "--quick" in sys.argv
    step("-m", "pytest", "-q")
    if not quick:
        step("run_all.py")
    for stage in ("prepare", "audit", "test"):
        step("-m", "experiments.final_system", "--stage", stage)
    for stage in ("base", "tabpfn"):
        step("-m", "experiments.forecast_model_search", "--stage", stage)
    for stage in ("tables", "fnfp", "peak"):
        step("-m", "src.final_analysis", "--stage", stage)
    step("-m", "src.report_figures")
    print("\n완료: outputs/final_system/test_predictions_regime_tabpfn.csv 가 최종 테스트 예측결과 파일이다.")


if __name__ == "__main__":
    main()
