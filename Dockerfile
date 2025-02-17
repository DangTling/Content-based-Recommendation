FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY endpoints.py .
COPY .env /app/.env

EXPOSE 5000
CMD ["python", "endpoints.py"]