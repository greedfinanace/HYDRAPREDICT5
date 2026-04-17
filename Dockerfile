FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/pyproject.toml
COPY quant_stack /app/quant_stack

RUN pip install --upgrade pip && pip install .

EXPOSE 8501

CMD ["streamlit", "run", "quant_stack/module9/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]

