FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY prod_assistant ./prod_assistant
RUN pip install --upgrade pip && pip install .
EXPOSE 8000
CMD ["uvicorn", "prod_assistant.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
