FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY titanic_model.py .

CMD ["python", "titanic_model.py"]

