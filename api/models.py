"""ML Model Manager - training, saving and loading models."""

import uuid
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from api.clearml_manager import get_clearml_manager

from clearml import Task as TaskClass, Task
import os
import concurrent.futures
import time

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ML model training, prediction, and metadata."""

    MODEL_TYPES = {
        "linear_regression": LinearRegression,
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
    }

    def __init__(self):
        self.model_metadata = {}

    def train(
        self,
        model_type: str,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        hyperparams: Optional[Dict] = None,
        log_to_clearml: bool = True,
    ) -> Dict:
        """Train model and return metrics."""
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_id = str(uuid.uuid4())[:8]
        hyperparams = hyperparams or {}
        model = self.MODEL_TYPES[model_type](**hyperparams)

        model.fit(X_train, y_train)

        metrics = {
            "train_r2": float(r2_score(y_train, model.predict(X_train))),
            "train_mse": float(mean_squared_error(y_train, model.predict(X_train))),
            "train_mae": float(mean_absolute_error(y_train, model.predict(X_train))),
        }

        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            metrics["test_r2"] = float(r2_score(y_test, y_pred))
            metrics["test_mse"] = float(mean_squared_error(y_test, y_pred))
            metrics["test_mae"] = float(mean_absolute_error(y_test, y_pred))

        model_info = {
            "model_id": model_id,
            "type": model_type,
            "hyperparams": hyperparams,
            "metrics": metrics,
            "created_at": datetime.now().isoformat(),
            "clearml_task_id": None,
            "stored_in_clearml": False,
        }

        if log_to_clearml:
            task_id = self._log_to_clearml(
                model_id, model_type, hyperparams, metrics, model=model
            )
            if not task_id:
                raise RuntimeError(
                    "Failed to store model in ClearML; aborting to enforce ClearML-only storage"
                )
            model_info["clearml_task_id"] = task_id
            model_info["stored_in_clearml"] = True

        self.model_metadata[model_id] = model_info

        return model_info

    def predict(self, model_id: str, X) -> list:
        """Make prediction with loaded model."""

        model = self._load_model(model_id)
        return model.predict(X)

    def retrain(
        self,
        model_id: str,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        hyperparams: Optional[Dict] = None,
        log_to_clearml: bool = True,
    ) -> Dict:
        """Retrain existing model with new data."""

        model_type = None
        if model_id in self.model_metadata:
            model_type = self.model_metadata[model_id].get("type")
        if model_type is None:
            raise ValueError(f"Model type for {model_id} unknown; cannot retrain")

        hyperparams = hyperparams or {}
        model = self.MODEL_TYPES[model_type](**hyperparams)
        model.fit(X_train, y_train)

        metrics = {
            "train_r2": float(r2_score(y_train, model.predict(X_train))),
            "train_mse": float(mean_squared_error(y_train, model.predict(X_train))),
            "train_mae": float(mean_absolute_error(y_train, model.predict(X_train))),
        }

        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            metrics["test_r2"] = float(r2_score(y_test, y_pred))
            metrics["test_mse"] = float(mean_squared_error(y_test, y_pred))
            metrics["test_mae"] = float(mean_absolute_error(y_test, y_pred))

        model_info = {
            "model_id": model_id,
            "type": model_type,
            "hyperparams": hyperparams,
            "metrics": metrics,
            "retrained_at": datetime.now().isoformat(),
            "clearml_task_id": None,
            "stored_in_clearml": False,
        }

        if log_to_clearml:
            task_id = self._log_to_clearml(
                model_id,
                model_type,
                hyperparams,
                metrics,
                task_suffix="retrain",
                model=model,
            )
            if not task_id:
                raise RuntimeError(
                    "Failed to store retrained model in ClearML; aborting to enforce ClearML-only storage"
                )
            model_info["clearml_task_id"] = task_id
            model_info["stored_in_clearml"] = True
        if model_id in self.model_metadata:
            self.model_metadata[model_id].update(model_info)
        else:
            self.model_metadata[model_id] = model_info

        return model_info

    def delete(self, model_id: str) -> None:
        """Delete model from memory and ClearML."""

        logger.info(f"Attempting to delete model {model_id} from ClearML")
        clearml = get_clearml_manager()
        if clearml.enabled:
            try:
                artifact_name = f"model_{model_id}"
                tasks = Task.get_tasks(project_name="ML-OPS-HW1")
                for task in tasks:
                    try:
                        if artifact_name in task.artifacts:
                            task.artifacts[artifact_name].delete()
                            logger.info(
                                f"Model {model_id} deleted from ClearML (task: {task.id})"
                            )
                            break
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(
                    f"Could not delete model from ClearML: {str(e)}", exc_info=True
                )

        if model_id in self.model_metadata:
            del self.model_metadata[model_id]

    def get_info(self, model_id: str) -> Dict:
        """Get model information from memory or ClearML."""

        if model_id in self.model_metadata:
            return {
                "model_id": model_id,
                "type": self.model_metadata[model_id].get("type"),
                "status": "registered",
                "stored_in_clearml": self.model_metadata[model_id].get(
                    "stored_in_clearml", False
                ),
                "clearml_task_id": self.model_metadata[model_id].get("clearml_task_id"),
            }

        clearml = get_clearml_manager()
        if clearml.enabled:
            logger.info(f"Model {model_id} not in memory, checking ClearML...")
            try:
                tasks = Task.get_tasks(project_name="ML-OPS-HW1")

                for task in tasks:
                    artifact_name = f"model_{model_id}"
                    if artifact_name in task.artifacts:
                        logger.info(
                            f"Found model {model_id} in ClearML (task: {task.id})"
                        )
                        return {
                            "model_id": model_id,
                            "type": task.artifacts[artifact_name].type,
                            "status": "stored_in_clearml",
                            "stored_in_clearml": True,
                            "clearml_task_id": task.id,
                        }
            except Exception as e:
                logger.error(
                    f"Error checking ClearML for model info: {str(e)}", exc_info=True
                )

        raise ValueError(f"Model {model_id} not found")

    def list_models(self) -> List[Dict]:
        """List all loaded models with storage information."""

        clearml = get_clearml_manager()
        results: List[Dict] = []
        if not clearml.enabled:
            return results
        try:
            tasks = Task.get_tasks(project_name="ML-OPS-HW1")
            seen = set()
            for task in tasks:
                for name, art in task.artifacts.items():
                    if name.startswith("model_"):
                        model_id = name.split("model_")[-1]
                        if model_id in seen:
                            continue
                        seen.add(model_id)
                        results.append(
                            {
                                "model_id": model_id,
                                "type": art.type,
                                "stored_in_clearml": True,
                                "clearml_task_id": task.id,
                            }
                        )
        except Exception:
            logger.exception("Failed to list models from ClearML")
        return results

    def _load_model(self, model_id: str):
        """Load model from memory cache or ClearML."""

        clearml = get_clearml_manager()
        if clearml.enabled:
            logger.info(
                f"Model {model_id} not in memory, attempting to load from ClearML..."
            )
            try:
                artifact_name = f"model_{model_id}"
                tasks = Task.get_tasks(project_name="ML-OPS-HW1")
                try:
                    MODEL_LOAD_TIMEOUT = float(
                        os.environ.get("MODEL_LOAD_TIMEOUT", "10")
                    )
                except Exception:
                    MODEL_LOAD_TIMEOUT = 10.0

                for task in tasks:
                    try:
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=1
                        ) as exe:
                            fut = exe.submit(
                                lambda a=task, name=artifact_name: a.artifacts[
                                    name
                                ].get()
                            )
                            try:
                                model = fut.result(timeout=MODEL_LOAD_TIMEOUT)
                            except concurrent.futures.TimeoutError:
                                logger.warning(
                                    f"Timeout loading model {model_id} from task {task.id}"
                                )
                                continue
                            except Exception as e:
                                logger.debug(
                                    f"Artifact get exception for model {model_id} task {task.id}: {e}"
                                )
                                continue

                        if model is not None:
                            logger.info(
                                f"Model {model_id} loaded from ClearML (task: {task.id})"
                            )
                            return model
                    except (KeyError, Exception):
                        continue

                logger.warning(f"Model {model_id} not found in ClearML artifacts")
            except Exception as e:
                logger.error(
                    f"Error loading model from ClearML: {str(e)}", exc_info=True
                )

        raise ValueError(
            f"Model {model_id} not found in ClearML. "
            f"Available models: {list(self.model_metadata.keys()) or 'none'}"
        )

    def _log_to_clearml(
        self,
        model_id: str,
        model_type: str,
        hyperparams: Dict,
        metrics: Dict,
        task_suffix: str = "train",
        model=None,
    ) -> Optional[str]:
        """Log model training to ClearML and save model object."""
        clearml = get_clearml_manager()
        if not clearml.enabled:
            logger.info(
                f"ClearML disabled, model {model_id} trained but not stored in ClearML"
            )
            return None

        try:
            task_name = f"{model_type}_{task_suffix}_{model_id}"
            task_id = clearml.init_task(task_name=task_name)

            if not task_id:
                logger.warning(f"Failed to initialize ClearML task for {model_id}")
                return None

            logger.info(f"ClearML task initialized: {task_id}")

            clearml.log_hyperparams(hyperparams)
            for metric_name, metric_value in metrics.items():
                clearml.log_scalar(
                    title="metrics", series=metric_name, value=metric_value
                )

            if model:
                artifact_name = f"model_{model_id}"
                clearml.upload_model_object(model, artifact_name)
                logger.info(
                    f"Model {model_id} saved to ClearML as primary storage: {artifact_name}"
                )

            clearml.close()
            return task_id
        except Exception as e:
            logger.error(f"ClearML logging error: {str(e)}", exc_info=True)
            logger.warning(
                f"Model {model_id} trained successfully but not saved to ClearML"
            )
            return None


_model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or create model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
