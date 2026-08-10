from datetime import datetime

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator


with DAG(
    dag_id = "fds_retrain",
    start_date = datetime(2026, 8, 1),
    schedule = "0 5 * * 1",  # every Monday at 5am
    catchup = False,
    # To prevent a race condition, we only allow one
    
    # catchup=False stops Airflow backfilling the whole history, but it still
    # fires the most recent missed slot the moment the DAG is unpaused. That is
    # how a manual trigger ended up racing a scheduled one, both writing the
    # same candidate file and doubling the training time. One run at a time.
    max_active_runs = 1,
    tags = ["fds", "retrain"]

) as dag:

    build = BashOperator(
        task_id="build_training_set",
        bash_command=(
            "cd /home/dhlim/fds && "
            "/home/dhlim/.local/bin/uv run python -m offline.build_training_set"
        ),
    )

    retrain = BashOperator(
        task_id="retrain_model",
        bash_command=(
            "cd /home/dhlim/fds && "
            "/home/dhlim/.local/bin/uv run python -m offline.train --out /home/dhlim/fds/data/model_candidate.txt"
        ),
    )

    promote = BashOperator(
        task_id="promote_model",
        bash_command=(
            "cd /home/dhlim/fds && "
            "/home/dhlim/.local/bin/uv run python -m offline.promote --candidate /home/dhlim/fds/data/model_candidate.txt"
        ),
    )

    build >> retrain >> promote