FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

RUN mkdir -p /app/model/
COPY model.joblib /app/model/model.joblib

EXPOSE 8080

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "app:app", "--timeout", "90"]
