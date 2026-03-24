import time
import vertexai
from pathlib import Path
from google import genai
from google.genai import types
from google.cloud import aiplatform
from typing import Any

from src.gemini_classification.config import (
    LOCATION,
    BUCKET_NAME,
    BUCKET_URI,
    PROJECT_ID,
    BASE_MODEL,
    DATASET_VERSION,
    SYSTEM_PROMPT_VERSION,
)
from src.gemini_classification.helpers import upload_directory_to_gcs


class VertexTuningManager:
    """Orchestrates GCS uploads and Supervised Fine-Tuning (SFT) jobs."""

    def __init__(self):
        # Initialize all SDKs once
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        aiplatform.init(
            project=PROJECT_ID, location=LOCATION, staging_bucket=BUCKET_URI
        )

        self.client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
        self.bucket_uri = BUCKET_URI
        self.base_model = BASE_MODEL

    def sync_data_to_gcs(
        self, local_data_path: str = "data/preprocessed_data", overwrite: bool = False
    ):
        """Ensures local processed data matches Cloud Storage."""
        print("--- Syncing local data to GCS ---")
        upload_directory_to_gcs(
            local_path=Path(local_data_path),
            bucket_name=BUCKET_NAME,
            project_id=PROJECT_ID,
            location=LOCATION,
            overwrite=overwrite,
        )

    def submit_sft_job(
        self,
        dataset_version: str,
        sys_prompt_v: int,
        balanced_data: bool,
        epochs: int = 5,
        adapter_size: str = "ADAPTER_SIZE_FOUR",
        learning_rate_multiplier: float = 1.0,
        wait: bool = False,
    ) -> Any:
        """Configures and submits a Supervised Fine-Tuning job."""

        version = dataset_version
        balanced_suffix = "_balanced" if balanced_data else ""

        # Consistent naming for tracking experiments
        job_display_name = (
            f"ft_{self.base_model.split('/')[-1]}_{version}_"
            f"ep{epochs}_lr{learning_rate_multiplier}_{adapter_size.lower()}_"
            f"sys{sys_prompt_v}{balanced_suffix}"
        )

        # Define GCS URIs based on the standard naming convention from our preprocessor
        train_uri = (
            f"{self.bucket_uri}/for_finetuning_gemini/{version}/"
            f"train_sys_prompt_{sys_prompt_v}_{version}{balanced_suffix}.jsonl"
        )
        val_uri = (
            f"{self.bucket_uri}/for_finetuning_gemini/{version}/"
            f"val_sys_prompt_{sys_prompt_v}_{version}{balanced_suffix}.jsonl"
        )

        print(f"Submitting Tuning Job: {job_display_name}")

        job = self.client.tunings.tune(
            base_model=self.base_model,
            training_dataset={"gcs_uri": train_uri},
            config=types.CreateTuningJobConfig(
                tuned_model_display_name=job_display_name,
                validation_dataset={"gcs_uri": val_uri},
                adapter_size=adapter_size,
                epoch_count=epochs,
                learning_rate_multiplier=learning_rate_multiplier,
            ),
        )
        if wait:
            self.wait_for_job(job.name)

        return job

    def wait_for_job(self, job_name, poll_interval: int = 5 * 60):
        """Waits for a TuningJob to finish."""
        terminal_states = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
        }

        tuning_job = self.client.tunings.get(name=job_name)
        while tuning_job.state not in terminal_states:
            print(f"Current state: {tuning_job.state}")
            time.sleep(poll_interval)
            tuning_job = self.client.tunings.get(name=job_name)

        print(f"Final tuning state: {tuning_job.state}")


if __name__ == "__main__":
    trainer = VertexTuningManager()

    # Upload the latest local files
    trainer.sync_data_to_gcs()

    # Grid search or multi-job submission
    training_configs = [
        {"v": DATASET_VERSION, "prompt": SYSTEM_PROMPT_VERSION, "bal": True},
        {"v": DATASET_VERSION, "prompt": SYSTEM_PROMPT_VERSION, "bal": False},
    ]

    for cfg in training_configs:
        trainer.submit_sft_job(
            dataset_version=cfg["v"],
            sys_prompt_v=cfg["prompt"],
            balanced_data=cfg["bal"],
        )
