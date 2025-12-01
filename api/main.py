from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import shutil
import os
import subprocess
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from api.dvc_manager import DVCManager
from api.models import get_model_manager
import asyncio
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.get("/status")
def status():
    """API health check."""
    return {"status": "ok"}


@app.get("/models/available")
def available_models():
    """Get list of available model types."""
    return {"models": ["linear_regression", "decision_tree", "random_forest"]}


UPLOAD_DIR = "data/datasets"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload dataset and version with DVC."""
    try:
        filepath = os.path.join(UPLOAD_DIR, file.filename)

        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        DVCManager.add_and_push(filepath)
        logger.info(f"Dataset uploaded and versioned: {file.filename}")
        return {"status": "uploaded", "file": file.filename}

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")


DATASETS_DIR = Path(UPLOAD_DIR)


@app.get("/datasets/download/{filename}")
def download_dataset(filename: str):
    """Download dataset from local storage or pull from S3 via DVC."""
    try:
        filepath = DATASETS_DIR / filename

        if not filepath.exists():
            logger.info(f"Dataset not in local storage, pulling from S3: {filename}")
            DVCManager.pull(str(filepath))

        if not filepath.exists():
            logger.warning(f"Dataset not found after pull attempt: {filename}")
            raise FileNotFoundError(f"Dataset {filename} not found in S3")

        return FileResponse(
            filepath, media_type="application/octet-stream", filename=filename
        )

    except FileNotFoundError as e:
        logger.error(f"File not found: {filename}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Download failed for {filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)}")


@app.post("/models/train")
async def train_model(
    model_type: str, dataset_path: str, hyperparams: Optional[Dict[str, Any]] = None
):
    """Train new ML model from dataset."""
    try:
        if not dataset_path:
            raise ValueError("dataset_path required")

        logger.info(
            f"Starting training: model_type={model_type}, dataset={dataset_path}"
        )

        filepath = DATASETS_DIR / dataset_path
        try:
            DVCManager.pull(str(filepath))
        except Exception as e:
            logger.warning(f"Could not pull dataset from S3, trying local: {e}")

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset {dataset_path} not found")

        df = pd.read_csv(filepath, sep=None, engine="python")

        if df is None or df.empty:
            raise ValueError(f"Dataset is empty: {dataset_path}")

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        split_idx = int(0.8 * len(df))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model_manager = get_model_manager()
        model_info = model_manager.train(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparams=hyperparams or {},
            log_to_clearml=True,
        )

        logger.info(f"Training completed: model_id={model_info['model_id']}")
        return model_info

    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Training failed: {str(e)}")


@app.post("/models/{model_id}/retrain")
async def retrain_model(
    model_id: str, dataset_path: str, hyperparams: Optional[Dict[str, Any]] = None
):
    """Retrain model with new dataset."""
    try:
        if not dataset_path:
            raise ValueError("dataset_path required")

        logger.info(f"Starting retrain: model_id={model_id}, dataset={dataset_path}")

        filepath = DATASETS_DIR / dataset_path
        try:
            DVCManager.pull(str(filepath))
        except Exception as e:
            logger.warning(f"Could not pull dataset from S3, trying local: {e}")

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset {dataset_path} not found")

        df = pd.read_csv(filepath, sep=None, engine="python")

        if df is None or df.empty:
            raise ValueError(f"Dataset is empty: {dataset_path}")

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        split_idx = int(0.8 * len(df))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model_manager = get_model_manager()
        model_info = model_manager.retrain(
            model_id=model_id,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparams=hyperparams or None,
            log_to_clearml=True,
        )

        logger.info(f"Retrain completed: model_id={model_id}")
        return model_info

    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Retrain failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Retrain failed: {str(e)}")


@app.post("/models/{model_id}/predict")
async def predict(model_id: str, data: Dict[str, Any] = None):
    """Make prediction with trained model."""
    try:
        logger.info(f"Prediction requested for model: {model_id}")

        if isinstance(data, dict):
            X = np.array([list(data.values())], dtype=np.float32)
        else:
            X = np.array(data, dtype=np.float32)

        model_manager = get_model_manager()
        loop = asyncio.get_running_loop()
        try:
            timeout = float(os.environ.get("MODEL_API_TIMEOUT", "120"))
        except Exception:
            timeout = 120.0

        try:
            predictions = await asyncio.wait_for(
                loop.run_in_executor(None, model_manager.predict, model_id, X),
                timeout=timeout,
            )

            logger.info(f"Prediction completed for model: {model_id}")
            return {
                "model_id": model_id,
                "prediction": (
                    predictions[0] if len(predictions) == 1 else predictions.tolist()
                ),
            }
        except asyncio.TimeoutError:
            logger.warning(
                f"Prediction timed out for model: {model_id} after {timeout}s"
            )
            raise HTTPException(
                status_code=504,
                detail=f"Prediction timed out after {int(timeout)} seconds",
            )

    except ValueError as e:
        logger.error(f"Model not found: {model_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/models")
async def list_models():
    """Get list of all trained models."""
    try:
        model_manager = get_model_manager()
        models = model_manager.list_models()
        logger.info(f"Listed {len(models)} models")
        return {"models": models}

    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to list models: {str(e)}")


@app.get("/models/{model_id}/info")
async def get_model_info(model_id: str):
    """Get information about specific model."""
    try:
        model_manager = get_model_manager()
        model_info = model_manager.get_info(model_id)
        logger.info(f"Retrieved info for model: {model_id}")
        return model_info

    except ValueError as e:
        logger.error(f"Model not found: {model_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400, detail=f"Failed to get model info: {str(e)}"
        )


@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete model."""
    try:
        model_manager = get_model_manager()
        model_manager.delete(model_id)
        logger.info(f"Model deleted: {model_id}")
        return {"status": "deleted", "model_id": model_id}

    except ValueError as e:
        logger.error(f"Model not found: {model_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to delete model: {str(e)}")


@app.get("/datasets")
async def list_datasets():
    """Get list of datasets from S3 via DVC."""
    try:
        datasets = DVCManager.list_s3_datasets()
        logger.info(f"Listed {len(datasets)} datasets")
        return {"datasets": datasets}

    except Exception as e:
        logger.error(f"Failed to list datasets: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400, detail=f"Failed to list datasets: {str(e)}"
        )


@app.delete("/datasets/{filename}")
async def delete_dataset(filename: str):
    """Delete dataset from local storage, DVC cache, and S3."""
    try:
        filepath = os.path.join(UPLOAD_DIR, filename)
        dvc_file = f"{filepath}.dvc"

        if not os.path.exists(filepath) and not os.path.exists(dvc_file):
            raise FileNotFoundError(f"Dataset {filename} not found")

        logger.info(f"Deleting dataset: {filename}")

        try:
            DVCManager.remove(filepath)
            logger.info(f"Dataset removed from DVC: {filename}")
        except Exception as e:
            logger.warning(f"DVC removal failed: {e}")

        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Local file deleted: {filename}")

        if os.path.exists(dvc_file):
            os.remove(dvc_file)
            logger.info(f"DVC metadata deleted: {filename}")

        try:
            subprocess.run(["git", "add", "-A"], check=False)
            subprocess.run(
                ["git", "commit", "-m", f"remove dataset {filename}"], check=False
            )
            logger.info(f"Git commit created for dataset removal: {filename}")
        except Exception as e:
            logger.warning(f"Git commit failed: {e}")

        logger.info(f"Dataset deleted successfully: {filename}")
        return {"status": "deleted", "filename": filename}

    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {filename}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete dataset: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=400, detail=f"Failed to delete dataset: {str(e)}"
        )
