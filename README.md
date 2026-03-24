# PneumOrpheus Inference Server

Dedicated FastAPI backend for inference APIs used by `pneumorpheus-app`.

## What this server provides

- `GET /cancer` service-status endpoint
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

- Provision (optional): creates resource group, ACR, App Service plan, and Web App on first run
- Validate: dependency install + Python compile check
- Containerize: builds and pushes image in ACR using `az acr build`
- Deploy (optional): deploys container to Azure Web App for Containers
- Pre-deploy validation for required variables / Key Vault secret

### Required pipeline variables for first deployment

- `azureSubscriptionServiceConnection`
- `webAppResourceGroup`
- `webAppName`
- `acrName`
- `appServicePlanName`
- `azureLocation`

Optional but recommended:

- `createInfrastructure=true` for initial run (set `false` after resources exist)
- `keyVaultName` and `keyVaultInferenceApiKeySecretName`

Recommended App Service settings:

- `WEBSITES_PORT=8001`
- `INFERENCE_API_KEY` (optional, pull from Key Vault)
- `MODEL_SOURCE=azure_blob` (or `local`)
- `AZURE_STORAGE_ACCOUNT_URL`, `AZURE_BLOB_CONTAINER`, `AZURE_BLOB_PREFIX`

Important: Azure Blob in this server setup is intended for model artifacts. Uploaded DICOM/NIfTI studies are not stored by this server in Azure Blob in the current flow.

### Model artifact layout in Blob

For `MODEL_SOURCE=azure_blob`, place artifacts under:

`<AZURE_BLOB_PREFIX>/<DEFAULT_MODEL_NAME>/<DEFAULT_MODEL_VERSION>/`

Example with your current settings:

`models/pneumorpheus-primary/v1/`

Recommended files:

- `model.pth` or `model.pt`
- `model_config.json` (optional but recommended)

Example `model_config.json`:

```json
{
	"model_file": "model.pth",
	"model_factory": "my_models.factory:create_model",
	"model_factory_kwargs": {
		"in_channels": 1,
		"out_channels": 3
	},
	"preprocessor_factory": "my_models.preprocess:from_study_bytes"
}
```

If no `preprocessor_factory` is configured, the runtime uses a zero-tensor fallback input shape from `MODEL_INPUT_SHAPE` to keep the inference path operational until full study preprocessing is integrated.

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
