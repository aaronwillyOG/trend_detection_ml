FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Upgrade pip and install with a longer timeout (1000s) to prevent disconnects
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]