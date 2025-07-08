FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt.serve .
RUN pip install -r requirements.txt.serve

COPY app.py .

EXPOSE 8080

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:8080", "app:app", "--timeout", "90"]
