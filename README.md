# PneumOrpheus Inference Server

Dedicated FastAPI backend for inference APIs used by `pneumorpheus-app`.

## What this server provides

- `GET /health` health endpoint
- `POST /infer` inference endpoint compatible with `pneumorpheus-app`
- `POST /v1/infer` versioned alias of the same endpoint
- Optional bearer token auth via `INFERENCE_API_KEY`

The multipart contract matches the app request shape:

- `analysisId`
- `patientId`
- `patientName`
- `modality`
- `clinicianEmail`
- `studyFile`

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

## Docker run

```bash
docker build -t pneumorpheus-inference-server .
docker run --rm -p 8001:8001 --env-file .env pneumorpheus-inference-server
```

## Azure DevOps

A dedicated pipeline is included in [azure-pipelines.yml](azure-pipelines.yml):

- Validate: dependency install + Python compile check
- Containerize: build/push to ACR
- Deploy (optional): deploy container to Azure Web App for Containers
- Pre-deploy validation for required variables / Key Vault secret

Recommended App Service settings:

- `WEBSITES_PORT=8001`
- `INFERENCE_API_KEY` (optional, pull from Key Vault)
- `MODEL_SOURCE=azure_blob` (or `local`)
- `AZURE_STORAGE_ACCOUNT_URL`, `AZURE_BLOB_CONTAINER`, `AZURE_BLOB_PREFIX`

## Model placement recommendation (production)

For your case (models currently in another location), the smartest production approach is:

1. **Store versioned model artifacts in Azure Blob Storage or Azure ML Model Registry**.
2. **Deploy this inference server separately** (yes, through Azure DevOps is the right choice).
3. **Use managed identity** from the inference app to pull model artifacts (avoid embedding storage keys).
4. **Pin model name/version via environment variables** for reproducibility.
5. **Cache model files on local disk/container volume** (`MODEL_CACHE_DIR`) to avoid repeated downloads.

### Why this is preferable

- Clear separation of concerns (UI/API app vs inference runtime)
- Independent scaling and rollout of model-serving container
- Safer secret handling with Key Vault + managed identity
- Better traceability of model versions tied to predictions

## Integration with pneumorpheus-app

Set the app environment variable:

```bash
INFERENCE_API_URL=https://<your-inference-service>/infer
```

If auth is enabled on this server, also set in app:

```bash
INFERENCE_API_KEY=<same bearer token>
```

## Where to plug real model inference

Replace the placeholder logic in:

- `app/services/model_runtime.py`
- `app/services/inference_service.py`

Keep response fields unchanged so the current app parser in `app/api/reports/route.ts` continues to work.
