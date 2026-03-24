from pathlib import Path

from google.cloud import aiplatform
from google.cloud import storage

from src.gemma_classification.config import (
    BASE_MODEL,
    BUCKET_NAME,
    BUCKET_URI,
    LOCATION,
    PROJECT_ID,
    HF_TOKEN 
)
from src.gemma_classification.helpers import upload_directory_to_gcs, download_gcs_file


class VertexTuningManager:
    """Handles GCS uploads and Vertex CustomJob submissions for Gemma 2B."""

    def __init__(self):
        # Initialize Vertex AI SDK
        aiplatform.init(
            project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI
        )
        self.bucket_uri = BUCKET_URI
        self.base_model = BASE_MODEL

    def sync_data_to_gcs(
        self, local_data_path: str = "data/preprocessed_data", overwrite: bool = False
    ):
        """Uploads preprocessed datasets to GCS."""
        print("--- Syncing local data to GCS ---")
        upload_directory_to_gcs(
            local_path=Path(local_data_path),
            bucket_name=BUCKET_NAME,
            project_id=PROJECT_ID,
            location=LOCATION,
            overwrite=overwrite,
        )

    def submit_custom_job(
        self,
        dataset_version: str,
        sys_prompt_v: int,
        balanced_data: bool,
        epochs: int = 3,
        lr: float = 1e-5,
    ):
        """Submits a Vertex CustomJob for fine-tuning Gemma 2B."""

        version = dataset_version
        balanced_suffix = "_balanced" if balanced_data else ""

        # Construct GCS paths for train, test and validation sets
        train_uri = (
            f"{self.bucket_uri}/for_finetuning_gemma/{version}/"
            f"train_sys_prompt_{sys_prompt_v}_{version}{balanced_suffix}.jsonl"
        )
        val_uri = (
            f"{self.bucket_uri}/for_finetuning_gemma/{version}/"
            f"val_sys_prompt_{sys_prompt_v}_{version}{balanced_suffix}.jsonl"
        )
        test_uri = (
            f"{self.bucket_uri}/for_testing_gemma/{version}/"
            f"test_sys_prompt_{sys_prompt_v}_{version}.jsonl"
        )

        # Job display name
        job_display_name = (
            f"ft_{self.base_model.split('/')[-1]}_{version}_"
            f"ep{epochs}_lr{lr}_sys{sys_prompt_v}{balanced_suffix}"
        )

        # Output directory for checkpoints
        output_dir = f"{self.bucket_uri}/gemma_checkpoints/{job_display_name}"

        print(f"Submitting CustomJob: {job_display_name}")
        print(f"Training data: {train_uri}")
        print(f"Validation data: {val_uri}")

        # Submit Vertex CustomJob
        job = aiplatform.CustomJob(
            display_name=job_display_name,
            worker_pool_specs=[
                {
                    "machine_spec": {
                        "machine_type": "g2-standard-8",
                        "accelerator_type": "NVIDIA_L4",
                        # "machine_type": "a2-ultragpu-1g",
                        # "accelerator_type": "NVIDIA_A100_80GB",
                        "accelerator_count": 1,

                    },  # adjust as needed, this configuration works
                    "replica_count": 1,
                    "container_spec": {
                        "env": [
                            {
                                "name": "HUGGING_FACE_HUB_TOKEN",
                                "value": f"{HF_TOKEN}"
                            }
                        ],
                        "image_uri": f"gcr.io/{PROJECT_ID}/gemma-classifier:latest",
                        "args": [
                            f"--train-data={train_uri}",
                            f"--val-data={val_uri}",
                            f"--output-dir={output_dir}",
                            f"--epochs={epochs}",
                            f"--lr={lr}",
                            f"--base_model={BASE_MODEL}",
                            f"--test-data={test_uri}"
                        ],
                    },
                }
            ],
        )

        # Run job synchronously
        job.run(sync=True)

        # Download the test predictions
        print("--- Downloading test predictions ---")
        
        gcs_pred_path = f"{output_dir}/test_predictions.txt"
        local_dir = Path("results/classification/submissions")
        local_filename = f"test_set_pred_{job_display_name}.txt"
        local_path = local_dir / local_filename
        
        download_gcs_file(gcs_pred_path, local_path, project_id=PROJECT_ID)

        return job
