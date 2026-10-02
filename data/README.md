# data

원본 `.mat` 파일(약 8 GB)은 용량 때문에 저장소에 포함하지 않습니다.

1. [Kaggle — Data-driven prediction of battery cycle life](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle)에서 아래 파일을 받아 이 폴더에 둡니다.

   | 파일 | 이 프로젝트에서의 이름 |
   |---|---|
   | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1 (학습) |
   | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2 (필수 Test) |
   | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3 (추가 평가, 선택 — 모델 선택에 미사용) |
   | `2018-04-03_varcharge_…mat` | 사용하지 않음 (가변 충전 실험) |

2. 프로젝트 루트에서 `python src/preprocess.py`를 실행하면 `data/processed/`에 셀 단위 표가 만들어집니다.

   | 파일 | 내용 |
   |---|---|
   | `cells.csv` | 셀 1행: 수명 라벨, 충전 정책, 사이클 ≤ 100에서 계산한 DAY 1 피처와 기록 오류를 정제한 `*_clean` 피처, EDA용 전체 수명 열화 지표 |
   | `qd_trajectories.csv` | 셀 × 사이클 방전 용량 Qd (EDA 그림용) |
   | `delta_q_curves.csv` | 셀별 ΔQ₁₀₀₋₁₀(V) 곡선, 1,000개 전압점 |

`.mat`은 MATLAB v7.3(HDF5) 형식이라 `h5py`로 필요한 배열만 읽습니다(전처리 약 5초).
