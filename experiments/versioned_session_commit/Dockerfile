FROM python:3.12.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1

WORKDIR /experiment

COPY experiments/versioned_session_commit/model.py ./
COPY experiments/versioned_session_commit/run_e31.py ./
COPY experiments/versioned_session_commit/decide_e31.py ./
COPY experiments/versioned_session_commit/test_model.py ./
COPY experiments/versioned_session_commit/README.md ./
COPY experiments/versioned_session_commit/run_docker.sh ./
COPY experiments/versioned_session_commit/Dockerfile ./

ENTRYPOINT ["python3"]
CMD ["run_e31.py", "--output-dir", "/results"]
