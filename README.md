# MLOps: Predicción de precios de casas con scikit-learn + Vertex AI serving




## Requisitos
- Python 3.8+
- GCP SDK configurado y autenticado
- Crear bucket en GCS para guardar modelos

```bash
PROJECT_ID=mlops-training-462812
REPO_NAME=docker-repository
IMAGE_NAME=house-price-trainer:latest
REGION=us-central1 
VERTEX_BUCKET=mlops-training-models-for-training-462812

gcloud auth login
gcloud auth application-default login --project mlops-training-462812
gcloud config set project mlops-training-462812 
```


## Subir el modelo a Vertex AI
```bash

gcloud ai models upload \
  --region=${REGION}  \
  --display-name="house-price-model-$(date +%Y%m%d%H%M%S)-crp" \
  --artifact-uri=gs://${VERTEX_BUCKET}/models/house-price-model/ \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest \
  --handler-path=predictor.py \
  --format="value(name)"

```


## Desplegar el modelo en un endpoint
```bash
YOUR_MODEL_ID=$(gcloud ai models list --region=${REGION} --filter="displayName:house-price-model" --format="value(name)" | cut -d '/' -f 6)
YOUR_ENDPOINT_DISPLAY_NAME="house-price-prediction-endpoint"
YOUR_ENDPOINT_ID_NUMERIC=$(gcloud ai endpoints list --region=${REGION} --filter="displayName:${YOUR_ENDPOINT_DISPLAY_NAME}" --format="value(name)" | cut -d '/' -f 6)

gcloud ai endpoints deploy-model ${YOUR_ENDPOINT_ID_NUMERIC} \
  --model=projects/${PROJECT_ID}/locations/${REGION}/models/${YOUR_MODEL_ID} \
  --display-name="house-price-deployment" \
  --machine-type=n1-standard-2 \
  --min-replica-count=1 \
  --max-replica-count=1 \
  --traffic-split=0=100 \
  --region=${REGION}
```

## Realizar Predicciones Online
```bash
PROJECT_ID=mlops-training-462812
REGION=us-central1
YOUR_ENDPOINT_ID_NUMERIC=$(gcloud ai endpoints list --region=${REGION} --filter="displayName:${YOUR_ENDPOINT_DISPLAY_NAME}" --format="value(name)" | cut -d '/' -f 6)

gcloud ai endpoints predict ${YOUR_ENDPOINT_ID_NUMERIC} \
  --region=${REGION} \
  --json-request=test_instance.json
```
