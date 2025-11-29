FROM python:3.10-slim
WORKDIR /api

RUN apt-get update && \
    apt-get install -y git curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY .git .git
COPY api/ api/
COPY data/ data/
COPY .dvc/ .dvc/
COPY dvc.yaml .
COPY .dvcignore .

RUN mkdir -p /root/.clearml

ENV CLEARML_HOME=/root/.clearml
ENV CLEARML_DISABLE_FAILED_CONNECTION_WARNING=true

RUN git config --global user.email "mlops@local" \
 && git config --global user.name "ml-api"

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
