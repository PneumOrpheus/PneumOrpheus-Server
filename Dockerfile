FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install server dependencies first (cached layer)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install the sclc inference package.
# sclc_pkg/ is populated by the CI pipeline (azure-pipelines.yml) before
# az acr build by copying sclc/ + pyproject.toml from the SCLC-Diagnostic repo.
# It is NOT tracked in git; see .gitignore.
COPY sclc_pkg/ ./sclc_pkg/
RUN pip install --no-cache-dir ./sclc_pkg

COPY app ./app

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
