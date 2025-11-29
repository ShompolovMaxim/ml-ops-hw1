"""ClearML integration for experiment logging."""

import os
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

home_dir = Path.home()
clearml_home = home_dir / ".clearml"
clearml_conf = clearml_home / "clearml.conf"

os.environ["CLEARML_HOME"] = str(clearml_home)

if clearml_conf.exists():
    os.environ["CLEARML_CONFIG_PATH"] = str(clearml_conf)
else:
    alt_conf = Path("/root/.clearml/clearml.conf")
    if alt_conf.exists():
        os.environ["CLEARML_CONFIG_PATH"] = str(alt_conf)
        os.environ["CLEARML_HOME"] = "/root/.clearml"

os.environ.setdefault("CLEARML_DISABLE_FAILED_CONNECTION_WARNING", "true")

from clearml import Task as TaskClass, Task
from clearml.backend_api.session.session import Session


class ClearMLManager:
    """Manager for ClearML integration."""
    
    def __init__(self):
        self.task: Optional[Task] = None
        self.enabled = Task is not None
    
    def init_task(self, task_name: str, project_name: str = "ML-OPS-HW1") -> Optional[str]:
        """Initialize ClearML task for logging."""

        if not self.enabled:
            logger.info("ClearML not available, task initialization skipped")
            return None
        logger.info(f"Initializing ClearML task: {task_name}")

        api_host = os.environ.get("CLEARML_API_HOST") or os.environ.get("CLEARML_API_SERVER")
        try:
            resp = requests.get(api_host, timeout=3)
            logger.info(f"ClearML API reachability check: {api_host} -> {resp.status_code}")
            server_reachable = resp.status_code < 500
        except Exception as conn_e:
            logger.warning(f"ClearML API not reachable ({api_host}): {conn_e}")
            server_reachable = False

        if not server_reachable:
            logger.warning("ClearML server not reachable, skipping SDK initialization")
            return None

        try:
            logger.info("Attempting online mode (Task.init)")
            self.task = TaskClass.init(
                project_name=project_name,
                task_name=task_name,
                auto_connect_frameworks={"scikit-learn": True},
                reuse_last_task_id=False
            )
            logger.info(f"✓ ClearML task created online: {self.task.id}")
            return self.task.id
        except Exception as e:
            logger.warning(f"✗ Online mode failed - {type(e).__name__}: {str(e)}")
            logger.info("Falling back to offline mode (Task.create)")

        try:
            logger.info("Attempting offline mode (Task.create)")
            self.task = TaskClass.create(
                project_name=project_name,
                task_name=task_name,
                task_type='training'
            )
            logger.info(f"✓ ClearML offline task created: {self.task.id}")
            return self.task.id
        except Exception as offline_e:
            logger.error(f"✗ Offline mode failed - {type(offline_e).__name__}: {str(offline_e)}")
            logger.warning("ClearML unavailable, continuing without task logging")
            return None
    
    def log_hyperparams(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters."""

        if not self.task:
            return
        try:
            self.task.connect_configuration(
                configuration_dict=hyperparams,
                name="hyperparameters"
            )
            logger.info(f"Hyperparameters logged: {len(hyperparams)} params")
        except Exception as e:
            logger.error(f"Error logging hyperparams: {str(e)}", exc_info=True)
    
    def log_scalar(self, title: str, series: str, value: float, iteration: int = 0) -> None:
        """Log scalar metric."""
        if not self.task:
            return
        
        try:
            self.task.get_logger().report_scalar(
                title=title,
                series=series,
                value=value,
                iteration=iteration
            )
            logger.debug(f"Metric logged: {title}/{series} = {value}")
        except Exception as e:
            logger.error(f"Error logging metric: {str(e)}", exc_info=True)
    
    def upload_model_object(self, model_object, model_name: str) -> None:
        """Upload model object as artifact."""
        if not self.task:
            return
        
        try:
            self.task.upload_artifact(
                name=model_name,
                artifact_object=model_object
            )
            logger.info(f"Model artifact uploaded: {model_name}")
        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}", exc_info=True)
    
    def download_model_object(self, model_name: str):
        """Download model object from artifacts."""
        if not self.task:
            logger.warning(f"ClearML not available, cannot download model: {model_name}")
            return None
        
        try:
            model = self.task.download_artifact(name=model_name)
            logger.info(f"Model artifact downloaded: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {str(e)}", exc_info=True)
            return None
    
    def close(self) -> None:
        """Close task."""
        if self.task:
            try:
                self.task.close()
                logger.info("ClearML task closed")
            except Exception as e:
                logger.error(f"Error closing task: {str(e)}", exc_info=True)


def get_clearml_manager() -> ClearMLManager:
    """Get ClearML manager instance."""
    return ClearMLManager()
