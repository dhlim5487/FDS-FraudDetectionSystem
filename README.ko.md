# FDS — Real-Time Fraud Detection System

한국어 | [English](README.md)

IEEE-CIS 사기 거래 데이터셋을 실시간 결제 스트림처럼 재현하고, 카드별 이력을 기반으로 피처를 계산해 LightGBM 모델로 거래를 스코어링했다. 오프라인 학습과 온라인 서빙이 **동일한 피처 계산 로직**을 공유하도록(train/serve parity) 설계했고, 주기적 재학습 → 검증 → 승격 → 서비스 반영까지 사람 손 없이 이어지도록 파이프라인을 만들었다.

## 아키텍처

```
train_transaction.csv ─┐
train_identity.csv     ├─▶ ingest/producer.py ─▶ Kafka ─▶ stream/consumer.py ─▶ Redis (카드별 이력)
                                                                                       │
                                                                                       ▼
                                                                        serving/app.py (FastAPI, /score)
                                                                                       ▲
                                                                    model.txt이 바뀌면 자동 재로드
                                                                       (재시작 없음)   │
offline/build_training_set.py ─▶ training_set.parquet ─▶ offline/train.py ─▶ model_candidate.txt
                                                                                       │
                                                             offline/promote.py (7가지 게이트) ┘
                                                                  통과 → model.txt 원자적 교체

Airflow DAG(orchestration/dags/retrain.py) — 매주 월요일 05:00 (0 5 * * 1), build → retrain → promote
   메타데이터 DB: Postgres (fds-postgres)
```

설계 원칙 (`core/schema.py`, `core/store.py`, `serving/app.py`):

- **이벤트는 실제 결제 단말이 보낼 법한 필드만 포함**한다. IEEE-CIS 원본 394개 컬럼 중 Vesta가 미리 가공해둔 V/C/D/M 계열은 쓰지 않고, `core/features.py`에서 직접 동등한 피처를 계산한다. 결제 단말이 실제로 관측할 수 있는 필드만 남겨 23개(거래 16 + 단말/식별 7)로 축소했다.
- **계산하는 곳(`core/features.py`)과 저장하는 곳(`core/store.py`)을 분리**해서, 오프라인 빌드는 `InMemoryStore`(~61,000 events/sec), 실시간 스트림은 `RedisStore`(~2,500 events/sec, 컨슈머 재시작에도 이력 유지)를 같은 인터페이스로 교체해 쓸 수 있게 했다.
- **`/score`는 읽기 전용**이다. 이력 기록은 `stream/consumer.py`만 담당한다. 스코어링 엔드포인트가 동시에 기록까지 하면 같은 거래가 두 번 기록되고, 다음 스코어링이 "일어난 적 없는 과거"를 기준으로 계산되는 문제가 생긴다.
- **피처를 캐시하지 않고 스코어링 시점에 재계산**한다. `amt_zscore_7d`, `is_new_device` 같은 피처는 지금 들어온 거래 자체(금액, 디바이스)에 좌우되므로, 저장된 피처를 그대로 쓰면 "이 거래"가 아니라 "이 카드의 마지막 거래"에 대한 답이 되어버린다.
- **모델 교체(승격)는 서비스 재시작 없이 반영**된다. `promote.py`는 임시 파일에 쓴 뒤 이름을 바꿔치는 방식(원자적 rename)으로 `model.txt`를 교체하고, `serving/app.py`는 요청마다 파일 수정시각만 확인해 바뀌었을 때만 다시 읽는다. 덮어쓰기가 아니라 rename이어야 하는 이유는, 읽는 쪽이 반쯤 쓰인 파일을 볼 가능성을 없애기 위해서다.

## 진행 단계 (Phase 1 → 6)

| Phase | 내용 |
|---|---|
| 1 | IEEE-CIS 데이터를 이벤트 시간 순으로 Kafka에 리플레이 |
| 2 | Redis 기반 피처 상태 저장 + train/serve parity 테스트 |
| 3 | LightGBM 모델 학습 + FastAPI 스코어링 서비스 (피처 10개, AUC 0.7235 / PR-AUC 0.1140) |
| 4 | 레이턴시 벤치마크 — p99 5.2ms @ ~360 req/sec |
| 5 | Airflow 재학습 DAG + 정량적 승격 게이트 (7종) |
| 6 | 피처 중요도 검증 후 스키마 확장 (피처 15개, **AUC 0.7513 / PR-AUC 0.1837**) |

### Phase 1 — 스트림 재현

`ingest/producer.py`가 `train_transaction.csv` + `train_identity.csv`를 이벤트 시간 순으로 정렬해 Kafka에 흘린다(`--speed`로 배속 조절). 이 단계에서 이벤트 스키마를 정했다: 원본 394개 컬럼 중 Vesta가 사후 가공한 V/C/D/M 계열을 버리고, 결제 단말이 실제로 관측할 수 있는 23개 필드(거래 16 + 단말/식별 7)만 남겼다.

### Phase 2 — 상태 저장과 parity

`core/features.py`(계산)와 `core/store.py`(저장)를 분리해, 오프라인은 `InMemoryStore`(~61,000 events/sec), 실시간은 `RedisStore`(~2,500 events/sec)를 같은 인터페이스로 바꿔 끼운다. 이력 기록은 `stream/consumer.py`만 담당한다. 두 경로가 정말 같은 값을 내는지는 `tests/test_parity.py`가 실제 거래 3,000건과 시간 경계 엣지 케이스로 고정한다.

### Phase 3 — 학습과 서빙

LightGBM(seed 42, 64 trees)을 시간 기준으로 분할해 학습(train 140일 472,432행 / test 42일 118,108행, 겹침 0). 피처 10개로 AUC 0.7235 / PR-AUC 0.1140. `serving/app.py`의 `/score`는 읽기 전용이고, 피처를 캐시하지 않고 요청마다 재계산한다.

### Phase 4 — 레이턴시 측정

`bench/latency`로 단일 워커 p99 5.2ms @ ~360 req/sec 확인. 워커를 4개로 늘렸을 때 오히려 p50이 44ms로 튀는 현상이 있었고, 원인은 자원 경쟁이 아니라 소켓 설정이었다(아래 "겪었던 오류" 2번).

### Phase 5 — 재학습 자동화

Airflow DAG(`orchestration/dags/retrain.py`)가 매주 월요일 05:00에 `build → retrain → promote`를 실행한다. `offline/promote.py`의 게이트 7종이 후보 모델을 정량 판정하며, 거부 경로와 승인 경로를 모두 실행해 확인했다. 수동 트리거와 스케줄 실행이 겹치는 문제는 `max_active_runs=1`로 막았다.

### Phase 6 — 스키마 확장

원본 CSV에서 스키마가 처음부터 버렸던 컬럼 전부(415개)를 다시 꺼내, 3단계로 기여도를 측정했다.

1. **통계 스크리닝** (`analysis/screen_columns.py`) — 결측률 / 준상수 여부 / 단변량 AUC·Cramér's V로 415개 컬럼 랭킹 (`analysis/out/screen_results.csv`)
2. **Gain importance + SHAP + Permutation importance** (`analysis/feature_contribution.py`) — 후보를 기존 10개 피처 위에 얹었을 때 실제로 성능이 오르는지 측정
3. **Null importance** (`analysis/null_importance.py`) — 라벨을 셔플해 80회 재학습, "우연히 중요해 보이는 수준"의 기준선을 만들고 실제 importance가 그 위인지 검증

세 단계를 다 통과한 건 `id_31`, `id_19`, `id_20`, `id_29`, `id_30` 다섯 개였고, 이걸 스키마에 넣어 AUC 0.7235 → 0.7513, PR-AUC 0.1140 → 0.1837. 반대로 기존 피처 중 `amt_sum_1h`(0.21), `txn_count_1h`(0.32), `amt_sum_24h`(0.41)는 노이즈 기준선 바로 위에 걸쳐 있어 "기여가 확실한 피처"는 아니다(지금은 남겨뒀다).

재학습 루프를 닫았다. 승격된 모델이 재시작 없이 반영되도록 `serving/app.py`에 mtime 기반 리로드를, `offline/promote.py`에 원자적 파일 교체를 넣고 실서버에서 왕복 검증했다(아래 "검증 방법" 참고). 정의만 되고 쓰이지 않던 Postgres에는 Airflow 메타데이터 DB 역할을 주어 SQLite에서 이전했다.

## 겪었던 오류와 해결 과정

작업하면서 실제로 막혔던 것들에 대한 원인과 조치를 적어둔다.

### 1. 카드별 최신 거래 캐시가 엉뚱한 답을 준 문제 (Phase 3)
`feat:<card1>`에 미리 계산해둔 피처를 캐시로 스코어링에 쓰려 했으나, 캐시된 피처는 "이 카드의 직전 거래" 기준이라 지금 채점 중인 거래와 어긋남. 캐시를 제거하고 스코어링 요청마다 `core/features.py`로 즉시 재계산하도록 바꿈.

### 2. `--workers 4`로 레이턴시가 오히려 급격히 나빠진 문제 (Phase 4)
동시성을 올리면 빨라질 거라 예상했지만 p50이 2.5ms → 44ms로 튐. 요청을 단계별로 나눠 측정해보니 서버 처리 시간(TTFB)은 그대로였고, 응답 헤더와 바디 사이에서 43ms가 사라짐.

| 구간 | 워커 1개 | 워커 4개 |
|---|---|---|
| TTFB (첫 바이트까지) | 3.87 ms | 4.06 ms |
| body (나머지 수신) | 0.06 ms | **43.38 ms** |

원인은 멀티프로세스 소켓 경로에 `TCP_NODELAY`가 빠져 있어, Nagle 알고리즘(작은 패킷을 모아 보냄)과 클라이언트의 지연 ACK(ACK를 모아 보냄)가 서로를 기다리다 40ms 타이머가 만료되었기 때문이었음.

실제 리눅스 서버에서 uvicorn 멀티 워커는 정상적인 관행이므로 이 결과를 일반화하지 않고, "이 환경(WSL2)에서 관측된 고정 지연"으로만 기록한다. (`OMP_NUM_THREADS=1` 쪽은 원인이 아니었고, 머신 자체의 ~30% 실행 편차 범위 안이었다.)

### 3. 승격 검증이 후보 모델이 아니라 자기 자신을 검증하고 있던 문제 (Phase 5)
게이트를 시험하려고 일부러 약한 모델(`--warmup-days 150`)을 만들어 넣었는데, 오히려 정상 모델(0.7235)을 크게 이긴 0.7443이 나왔다. 원인은 `promote.py`가 후보를 채점할 때 **후보가 실제로 학습한 구간이 아니라 스크립트가 새로 만든 표준 분할**을 시험지로 썼기 때문이다. 약한 모델의 학습 구간(day 150~182)이 표준 시험지(day 140~182) 안에 통째로 들어가 있었다. 0.7443은 후보가 이미 본 데이터를 다시 채점한 점수였다.

"train/test 겹침 없음" 게이트를 추가하니 시험지 118,108건 중 99,622건(84%)을 후보가 이미 학습했다고 걸러냈다. 겹치거나 아니거나 둘 중 하나라서 임계값을 정할 것도 없고, 멀쩡한 모델을 잘못 기각할 일도 없다.

이후 Airflow 상에서 3개 태스크가 모두 성공하고 게이트가 개선폭 +0.0000을 이유로 승격을 거부하는 것, 그리고 임계값을 임시로 낮췄을 때 실제로 `PROMOTED`로 파일이 교체되는 것까지 양쪽 분기를 모두 확인했다.

### 4. Airflow `catchup=False`인데도 수동 트리거가 스케줄 실행과 경합한 문제 (Phase 5)
`catchup=False`는 과거 전체를 백필하지 않을 뿐, DAG를 unpause하는 순간 가장 최근에 놓친 스케줄은 그대로 실행한다. 이 때문에 수동 트리거가 스케줄 실행과 겹쳐 같은 후보 파일에 동시에 쓰면서 학습 시간이 두 배로 걸리는 레이스 컨디션이 발생했다. `max_active_runs=1`로 동시 실행을 1개로 제한해 해결했다.

### 5. 첫 분석에서 "실시간에 존재할 수 없는 데이터"로 성능을 올려버린 문제 (Phase 6)
`feature_contribution.py` 1차 실행은 V/C/D/M 컬럼까지 전부 후보로 넣었고, AUC가 0.7235 → **0.8806**까지 올랐다. 그런데 이 컬럼들은 Vesta가 사후에 가공해둔 값이라 결제 단말이 실시간으로 보낼 수 있는 데이터가 아니다. 위에 적어둔 설계 원칙을 정작 분석 단계에서 어기고 있었다.

`--mode full`(참고용 상한 벤치마크)과 `--mode realistic`(단말이 보낼 수 있는 것만)으로 모드를 분리해 다시 돌렸고, 실제 스키마 확장은 realistic 결과(AUC 0.7578)만 근거로 진행했다. 0.8806은 "이 데이터셋에서 도달 가능한 상한"으로만 남겨뒀다.

### 6. 분석 스크립트가 두 번 크래시 (Phase 6)
- **1차**: SHAP dependence plot 저장 단계에서 죽음 (Stage 3 끝나기 직전)
- **2차**: permutation importance 진입 직후 죽음 — sklearn의 `permutation_importance`가 `.fit` 메서드를 요구하는데 raw LightGBM `Booster` 객체에는 없어서 거부됨 - 작은 래퍼 클래스로 감싸 해결함.

두 번 다 마지막 단계까지 완주한 적이 없어서, 세 번째 실행이 17분 넘게 걸릴 때 "멈춘 것"과 "원래 오래 걸리는 것"을 구분하지 못했음. `n_repeats=10 × 후보 128개 = 1,280번`의 재예측이 정상 소요 시간이었음.

### 7. id_ 피처 5개를 추가하면서 3곳이 동시에 깨진 문제 (Phase 6)
스키마 확장은 `core/schema.py`(계약) → `core/features.py`(계산) → `ingest/producer.py`(입력) 세 파일을 같이 고쳐야 하는데, 하나씩 빠뜨릴 때마다 다른 지점에서 터졌다.

- `core/features.py`의 `NameError`
- `ingest/producer.py`가 `train_identity.csv`를 아예 안 읽고 있었음 → id_ 필드가 전부 None으로 흘러감
- `core/features.py`에서 float/str 타입이 섞여 나가 **59만 행을 다 만든 뒤 마지막 `to_parquet`에서 pyarrow 크래시**(57초 낭비). 결측을 `"__missing__"` 문자열로 통일해 해결.

`tests/test_parity.py`도 같은 변경으로 두 번 깨졌다: `ImportError: cannot import name 'id_31' from 'core.schema'` → 고치니 `NameError: name 'id_31' is not defined`(`evt()` 헬퍼 시그니처에 파라미터를 안 넣었음).

### 8. `data/model.txt`가 조용히 덮어써진 문제 (Phase 6)
`offline/train.py`의 기본 출력 경로가 `data/model.txt`라, 검증용으로 재학습 한 번 돌린 것만으로 기존 10피처 프로덕션 모델과 `model.meta.json`이 교체됐다. 비교 기준으로 쓰던 baseline이 파일로는 남지 않았다. 이후로는 후보 학습에 `--out data/model_candidate.txt`를 반드시 쓴다.

### 9. `promote.py`에 범주형 처리가 빠져 있던 문제 + 중복 로직
`offline/promote.py`가 새로 추가된 5개 id_ 필드의 범주형 처리 로직을 갖고 있지 않아 오류가 났고, `offline/train.py`와 `analysis/null_importance.py`에도 비슷한 로직이 각각 중복돼 있었다. `core/features.py`에 `apply_categorical_dtype()`을 하나 만들어 세 곳이 공유하도록 정리했다.

### 10. 그 통합에서 호출부 하나를 놓쳐 `/score`가 죽어 있던 문제
9번에서 헬퍼로 통합할 때 `train.py` / `promote.py` / `null_importance.py`는 고쳤는데 **`serving/app.py`를 빠뜨렸다.** 모델은 id_ 5개를 범주형으로 학습했는데 채점할 때는 평범한 값으로 넘기고 있어서, 요청이 들어오는 즉시 죽는 상태였다.

```
ValueError: train and valid dataset categorical_feature do not match.
```

parity 테스트 3개는 계속 통과했고(피처 계산까지만 검사한다), 스키마를 확장한 뒤로 서버를 안 켜봤기 때문에 인지하지 못했다. 위 "재학습 루프 닫기"를 검증하려고 uvicorn을 띄운 순간 첫 요청에서 드러났다.

이후로는 헬퍼로 묶기 전에 `grep -rn`으로 호출부를 먼저 세고, 스키마를 건드린 날은 pytest만 보지 않고 uvicorn을 띄워 `/score`에 요청을 한 번 넣어본다.

### 환경 설정에서 막혔던 것들

깊이 있는 문제는 아니지만 실제로 시간을 쓴 것들:

- **`ModuleNotFoundError: No module named 'core'`** — `python ingest/producer.py`로 실행해서 생긴 문제. 패키지 상대 임포트를 쓰려면 `python -m ingest.producer` 형태여야 한다.
- **WSL 파일 소유자가 root** — 컨테이너/sudo를 거쳐 생성된 파일이 root 소유가 되어, VSCode-WSL에서 저장할 때 `EACCES: permission denied`. `chown`으로 되돌리고 컨테이너를 비-root 사용자로 실행하도록 수정했다.
- **Docker Desktop의 WSL 통합 비활성화** — WSL 쪽은 `The command 'docker' could not be found in this WSL 2 distro`, PowerShell 쪽은 `npipe:////./pipe/dockerDesktopLinuxEngine` 접속 실패. Docker Desktop 설정에서 WSL integration을 켜야 한다.
- **컨테이너 이름 / 스크립트 경로** — `docker exec -it kafka ...`는 `No such container: kafka`. 실제 이름은 `fds-kafka`이고, CLI도 `/opt/kafka/bin/kafka-console-consumer.sh`처럼 전체 경로와 확장자가 필요하다.
- **Airflow standalone 첫 기동 실패** — `Exception in thread scheduler / dag-processor / triggerer / api-server`로 전부 죽음. `AIRFLOW__CORE__DAGS_FOLDER`를 잡아준 뒤 재기동했다. 종료할 때도 `pkill -f`와 SIGTERM이 안 먹어서 `pkill -9`로 8080 포트를 비워야 했다.
- **Airflow 3의 `dags list`는 폴더가 아니라 DB를 본다** — Postgres로 옮긴 직후 DAG 목록이 텅 비어 나왔다. Airflow 2 습관대로 "폴더에 파일이 있으니 보이겠지"라고 생각하면 헤맨다. `airflow dags reserialize`로 폴더를 파싱해 DB에 등록시켜야 한다.
- **`load_examples = False`로 껐는데 CLI가 죽음** — 이미 DB에 등록된 예제 DAG 113개가 남아 `DeserializationError: ... 'example_custom_weight'`로 `dags list`가 통째로 실패했다. 예제를 끄면 그 DAG가 참조하던 클래스도 같이 사라지기 때문이다. stale로 표시하는 것만으로는 부족해서, `fileloc`이 Airflow 패키지 내부 `example_dags/`인 행만 골라 삭제했다.

## 알려진 한계

- **스트림 경로와 학습 경로가 데이터로 이어져 있지 않음** — `offline/build_training_set.py`는 Kafka로 흘린 이벤트가 아니라 원본 CSV를 다시 읽는다. 두 경로가 같은 피처 코드를 쓴다는 건 parity 테스트로 증명했지만, 이벤트를 Parquet/Postgres에 쌓아두고 그걸로 학습하는 콜드 패스는 만들지 않았다.
- **`offline/promote.py`의 데이터 위생 임계값 2개**(학습 행 수 하한, 사기율 허용 범위)는 아직 실측 근거 없이 코드상 판단값(judgement call)으로 표시돼 있다. 실제 주간 재학습이 몇 번 쌓여야 근거를 만들 수 있다.
- **Null importance 하위 3개 피처**(`amt_sum_1h`, `txn_count_1h`, `amt_sum_24h`)는 노이즈 기준선을 겨우 넘는 수준이라, 제거했을 때 성능이 유지되는지 확인이 필요하다.
- **Redis는 영속화를 꺼둔 상태**(`--save "" --appendonly no`)라 컨테이너를 내리면 카드별 이력이 사라진다. 리플레이로 다시 채우면 되는 데모 환경이라 그대로 뒀다.

## 실행 방법

### 1. 인프라 기동 (Kafka / Redis / Postgres)
```bash
docker compose up -d
```

### 2. 오프라인 학습 데이터 생성 → 모델 학습
```bash
uv run python -m offline.build_training_set     # data/training_set.parquet 생성
uv run python -m offline.train                  # data/model.txt 생성 (AUC/PR-AUC 출력)
```

### 3. 실시간 스트림 재생 + 컨슈머 기동
```bash
uv run python -m ingest.producer --speed 1000     # 거래를 이벤트 시간 배속으로 Kafka에 재생
uv run python -m stream.consumer                  # Kafka 소비 → Redis에 이력 적재 → feat:<card1> 갱신
```

### 4. 스코어링 서비스
```bash
uv run uvicorn serving.app:app --port 8000
curl -X POST localhost:8000/score -H 'content-type: application/json' -d '{...}'
```

### 5. 레이턴시 벤치마크
```bash
uv run python -m bench.latency --requests 10000
```

### 6. 재학습 → 승격 게이트 (수동 실행)
```bash
uv run python -m offline.build_training_set
uv run python -m offline.train --out data/model_candidate.txt
uv run python -m offline.promote --candidate data/model_candidate.txt
```
게이트를 통과하면 `data/model.txt`가 원자적으로 교체되고, 돌고 있는 스코어링 서비스가 다음 요청에서 알아서 새 모델을 집는다.

### 7. Airflow 재학습 DAG
```bash
airflow standalone                       # 설정은 airflow.cfg에 있어 export 불필요
airflow dags unpause fds_retrain         # 데모할 때만
```
매주 월요일 05:00에 `build → retrain → promote` 순으로 실행된다. 메타데이터는 `fds-postgres` 컨테이너에 저장된다.

### 8. 모델 검증 (Phase 6)
```bash
uv run python -m analysis.screen_columns
uv run python -m analysis.feature_contribution --mode realistic   # --mode full 은 참고용 상한
uv run python -m analysis.null_importance --runs 80
```

### 9. 테스트
```bash
uv run pytest tests/ -v      # parity 3개 + 모델 재로드 3개 (Redis 필요)
```

## 프로젝트 구조

```
core/         이벤트 스키마, 피처 계산, 이력 저장소 (InMemory/Redis)
ingest/       IEEE-CIS CSV → Kafka 리플레이
stream/       Kafka 컨슈머 (이력 적재 + 최신 피처 발행)
offline/      학습 데이터 빌드, 학습, 승격 게이트
serving/      FastAPI 스코어링 서비스 (모델 자동 재로드)
orchestration/ Airflow 재학습 DAG
analysis/     피처 중요도 검증 (SHAP, permutation, null importance 등)
bench/        레이턴시 벤치마크
tests/        train/serve parity + 모델 리로드 테스트
```

## 검증 방법

- **train/serve parity** — 실제 거래 3,000건을 오프라인(메모리) 경로와 온라인(Redis) 경로로 각각 계산해 소수점까지 일치하는지 비교. 경계 조건(1시간/24시간/7일 경계 정각, 동일 시각 거래, 카드 간 이력 격리)은 별도 엣지 케이스로 고정.
- **미래 정보 누수** — 어떤 피처도 자기 자신 이후의 거래를 보지 않는지 검사.
- **모델 재로드** — uvicorn을 켜둔 채 완전히 동일한 요청을 반복하면서 `promote.py`의 `install()`로 모델을 갈아끼웠다. 재시작 없이 점수가 `0.0198 → 0.0197`로 바뀌었고, 원본을 되돌리자 `0.0198`로 복귀했다. 단위 테스트는 "안 바뀌면 재로드하지 않는 것"까지 함께 고정한다(그게 없으면 매 요청 400KB를 다시 읽는 코드로 퇴화해도 통과한다).
- **승격 게이트** — 거부 경로(개선폭 +0.0000)와 승인 경로(임계값을 임시로 낮춰 `PROMOTED` 확인) 양쪽을 모두 실행.

## 성능 지표

현재 프로덕션 모델 (`data/model.meta.json` — 피처 15개, seed 42, 64 trees):

| 지표 | 값 | 의미 |
|---|---|---|
| AUC | **0.7513** | 0.5 = 의미 없이 찍는 수준 |
| PR-AUC | **0.1837** | 무작위 추측 0.0344 대비 5.3배 |
| recall@fpr0.001 | 0.0381 | 정상 1,000건 중 1건만 오탐할 때의 탐지율 |
| recall@fpr0.01 | 0.1442 | 정상 100건 중 1건 오탐 허용 시 탐지율 |
| prec@top0.5pct | 0.4644 | 위험도 상위 0.5%만 차단할 때의 정밀도 |
| prec@top1pct | 0.3810 | 상위 1% 차단 시 정밀도 |

- 학습/시험 분할: 시간 기준 (train 472,432행 140일 / test 118,108행 42일, 겹침 0)
- Phase 3의 10피처 모델은 AUC 0.7235 / PR-AUC 0.1140이었고, Phase 6의 스키마 확장으로 각각 +0.028 / +0.070 개선됨
- Vesta 사전 가공 컬럼까지 전부 넣으면 AUC 0.8806까지 오르지만, 실시간 재현이 불가능해 채택하지 않음 (위 "겪었던 오류" 5번)

그 외:

- 레이턴시: p99 5.2ms @ ~360 req/sec (단일 워커)
- 피처 저장소: InMemoryStore ~61,000 events/sec (오프라인) / RedisStore ~2,500 events/sec (실시간, 재시작에도 이력 유지)

## 요구 사항

- Python >= 3.11 (uv 사용)
- Docker / Docker Compose (Kafka, Redis, Postgres)
- 주요 의존성: `confluent-kafka`, `fastapi`, `lightgbm`, `pandas`, `pyarrow`, `redis`, `scikit-learn`, `uvicorn` (dev: `matplotlib`, `pytest`, `shap`)
- Airflow는 `uv tool`로 프로젝트와 분리 설치하고, 메타데이터 DB용 드라이버를 함께 넣는다:
  ```bash
  uv tool install apache-airflow --with psycopg2-binary --with asyncpg
  ```
  (`asyncpg`는 Airflow 3의 API 서버가 비동기 SQLAlchemy를 쓰기 때문에 필요)
