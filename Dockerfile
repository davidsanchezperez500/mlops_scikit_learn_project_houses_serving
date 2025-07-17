FROM us-central1-docker.pkg.dev/mlops-training-462812/docker-repository/house-price-base:latest

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY app.py .

RUN mkdir -p /app/model/
COPY model.joblib /app/model/model.joblib

EXPOSE 8080

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "app:app", "--timeout", "90"]
