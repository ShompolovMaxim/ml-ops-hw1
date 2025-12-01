"""DVC integration for dataset versioning."""

import subprocess
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class DVCManager:
    """Manages DVC versioning for datasets."""

    @staticmethod
    def add_and_push(filepath: str):
        """Add file to DVC, commit to git, and push to S3."""
        try:
            logger.info(f"Adding file to DVC: {filepath}")
            subprocess.run(["dvc", "add", filepath], check=True)
            subprocess.run(["git", "add", f"{filepath}.dvc"], check=True)
            subprocess.run(
                ["git", "commit", "-m", f"add dataset {Path(filepath).name}"],
                check=True,
            )
            subprocess.run(["dvc", "push"], check=True)

            os.remove(filepath)
            logger.info(f"Local file removed (stored in S3): {filepath}")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"DVC add_and_push failed for {filepath}: {str(e)}", exc_info=True
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error in add_and_push: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def pull(filepath: str):
        """Pull file from S3 via DVC."""
        try:
            logger.info(f"Pulling file from S3: {filepath}")
            subprocess.run(["dvc", "pull", filepath], check=True)
            logger.info(f"Successfully pulled: {filepath}")
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC pull failed for {filepath}: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in pull: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def remove(filepath: str):
        """Remove file from DVC and S3."""
        try:
            logger.info(f"Removing file from DVC: {filepath}")
            subprocess.run(["dvc", "remove", filepath], check=True)

            logger.info(f"Pushing removal to S3: {filepath}")
            subprocess.run(["dvc", "push"], check=True)
            logger.info(f"Successfully removed from DVC and S3: {filepath}")
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC remove failed for {filepath}: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in remove: {str(e)}", exc_info=True)
            raise

    @staticmethod
    def list_s3_datasets():
        """List datasets from .dvc files."""
        try:
            datasets = []
            dvc_dir = Path("data/datasets")

            if dvc_dir.exists():
                logger.info(f"Scanning for datasets in {dvc_dir}")
                for dvc_file in dvc_dir.glob("*.dvc"):
                    dataset_name = dvc_file.stem
                    try:
                        with open(dvc_file, "r") as f:
                            dvc_content = f.read()
                        datasets.append(dataset_name)
                        logger.debug(f"Found dataset: {dataset_name}")
                    except Exception as e:
                        logger.error(
                            f"Error reading {dvc_file}: {str(e)}", exc_info=True
                        )

                logger.info(f"Total datasets found: {len(datasets)}")
            else:
                logger.warning(f"Datasets directory not found: {dvc_dir}")

            return datasets
        except Exception as e:
            logger.error(f"Error listing S3 datasets: {str(e)}", exc_info=True)
            return []
