FROM python:3.11-slim

# System dependencies required by Cartopy / GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p cache

# Hugging Face Spaces uses port 7860 by default
EXPOSE 7860

ENV PORT=7860

CMD ["python", "main.py"]
