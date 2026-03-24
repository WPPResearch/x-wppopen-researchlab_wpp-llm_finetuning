import time
from typing import Any, Dict, List, Optional, Union

import vertexai
from google import genai
from google.cloud import aiplatform
from vertexai.tuning import sft

from src.gemini_classification.config import (
    BASE_MODEL,
    BUCKET_URI,
    DATASET_VERSION,
    LOCATION,
    PROJECT_ID,
    SYSTEM_PROMPT_VERSION,
)


class VertexBatchInferenceManager:
    """Manages model retrieval and batch prediction job submission on Vertex AI."""

    def __init__(self, project_id: str = PROJECT_ID, location: str = LOCATION):
        self.project_id = project_id
        self.location = location

        # Initialize SDKs
        vertexai.init(project=self.project_id, location=self.location)
        aiplatform.init(project=self.project_id, location=self.location)
        self.client = genai.Client(
            vertexai=True, project=self.project_id, location=self.location
        )

    def _generate_display_name(
        self,
        dataset_version: str,
        sys_prompt_v: int,
        balanced_data: bool,
        n_epochs: int,
        adapter_size: str,
        lrm: Union[int, float],
    ) -> str:
        """Reproduces the naming convention used during training."""
        version = dataset_version
        balanced = "_balanced" if balanced_data else ""
        model_id = BASE_MODEL
        return f"ft_{model_id}_{version}_ep{n_epochs}_lr{lrm}_{adapter_size.lower()}_sys{sys_prompt_v}{balanced}"

    def get_tuned_model_metadata(self, display_name: str) -> Optional[Dict[str, Any]]:
        """Searches for a completed SFT job and returns its metadata."""
        jobs = sft.SupervisedTuningJob.list(filter=f'display_name="{display_name}"')

        if not jobs:
            print(f"No job found with display name: {display_name}")
            return None

        # Get the most recent job
        tuning_job = jobs[0]
        print(f"Retrieved job: {tuning_job.resource_name}")
        return tuning_job.to_dict()

    def run_batch_inference_on_base_model(
        self,
        dataset_version: str,
        sys_prompt_v: int,
        sets_to_run: List[str] = ["val", "test"],
        wait: bool = False,
    ):
        """Orchestrates batch predictions for specified data splits using the non-finetuned base model to serve as control."""
        display_name = self._generate_display_name(
            dataset_version,
            sys_prompt_v,
            balanced_data=False,
            n_epochs=0,
            adapter_size="NA",
            lrm=0,
        )
        model_resource_name = BASE_MODEL
        # Define output Enum based on prompt version
        response_enum = (
            ["0", "1", "2"]
            if sys_prompt_v == 3
            else ["Overperforming", "Underperforming", "Average performance"]
        )

        for split in sets_to_run:
            if split == "val":
                data_uri = f"gs://ds_finetuning_for_campaign_outcomes/for_evaluation_gemini/{dataset_version.upper()}/val_sys_prompt_{sys_prompt_v}_{dataset_version.upper()}.jsonl"

            elif split == "test":
                data_uri = f"gs://ds_finetuning_for_campaign_outcomes/for_testing_gemini/{dataset_version.upper()}/test_sys_prompt_{sys_prompt_v}_{dataset_version.upper()}.jsonl"
            else:
                print("Supported splits are only test and val.")
                return
            # GCS Destination
            job_name = f"{split}_set_pred_baseline_{model_resource_name}_sys_prompt{sys_prompt_v}_{dataset_version.upper()}"
            dest_path = f"{BUCKET_URI}/batch_predictions/{job_name}".replace("-", "_")

            print(f"Submitting {split} inference for {job_name}...")
            jobs = []
            batch_job = aiplatform.BatchPredictionJob.submit(
                model_name=f"publishers/google/models/{model_resource_name}",
                job_display_name=job_name,
                gcs_source=data_uri,
                gcs_destination_prefix=dest_path,
                instances_format="jsonl",
                predictions_format="jsonl",
                model_parameters={
                    # "temperature": 0.0,
                    "response_mime_type": "text/x.enum",
                    "response_schema": {"type": "string", "enum": response_enum},
                },
            )
            print(f"Job submitted: {batch_job.resource_name}")
            jobs.append(batch_job)

        if wait:
            self.wait_for_jobs(jobs)

    def run_batch_inference(
        self,
        dataset_version: str,
        sys_prompt_v: int,
        balanced_data: bool,
        n_epochs: int = 5,
        adapter_size: str = "ADAPTER_SIZE_FOUR",
        lrm: Union[int, float] = 1,
        sets_to_run: List[str] = ["val", "test"],
        wait: bool = False,
    ) -> None:
        """Orchestrates batch predictions for specified data splits."""

        display_name = self._generate_display_name(
            dataset_version, sys_prompt_v, balanced_data, n_epochs, adapter_size, lrm
        )

        metadata = self.get_tuned_model_metadata(display_name)
        if not metadata:
            return

        model_resource_name = metadata["tunedModel"]["model"]

        # Define output Enum based on prompt version
        response_enum = (
            ["0", "1", "2"]
            if sys_prompt_v == 3
            else ["Overperforming", "Underperforming", "Average performance"]
        )

        for split in sets_to_run:
            data_uri = self._get_data_uri(
                split, metadata, sys_prompt_v, dataset_version
            )

            job_name = metadata["tunedModelDisplayName"].replace(
                "ft_", f"{split}_set_pred_"
            )
            # GCS Destination
            dest_path = f"{BUCKET_URI}/batch_predictions/{job_name}".replace("-", "_")

            print(f"Submitting {split} inference for {job_name}...")
            jobs = []
            batch_job = aiplatform.BatchPredictionJob.submit(
                model_name=model_resource_name,
                job_display_name=job_name,
                gcs_source=data_uri,
                gcs_destination_prefix=dest_path,
                instances_format="jsonl",
                predictions_format="jsonl",
                model_parameters={
                    "temperature": 0.0,
                    "response_mime_type": "text/x.enum",
                    "response_schema": {"type": "string", "enum": response_enum},
                },
            )
            print(f"Job submitted: {batch_job.resource_name}")
            jobs.append(batch_job)

        if wait:
            self.wait_for_jobs(jobs)

    def _get_data_uri(
        self, split: str, metadata: Dict[str, Any], sys_v: int, version: str
    ) -> str:
        """Logic to switch between training, evaluation, and testing GCS paths."""
        spec = metadata["supervisedTuningSpec"]

        if split == "train":
            return spec["trainingDatasetUri"].replace(
                "for_finetuning_gemini", "for_evaluation_gemini"
            )

        if split == "val":
            return spec["validationDatasetUri"].replace(
                "for_finetuning_gemini", "for_evaluation_gemini"
            )

        # Test sets live in a different directory structure
        val_uri = spec["validationDatasetUri"]
        base_path = val_uri.split("/for_finetuning_gemini/")[0]
        return f"{base_path}/for_testing_gemini/{version}/test_sys_prompt_{sys_v}_{version}.jsonl"

    def wait_for_jobs(
        self, jobs: List[aiplatform.BatchPredictionJob], poll_interval: int = 5 * 60
    ):
        """Waits for multiple BatchPredictionJobs to finish by polling their state."""

        print("Waiting for all batch inference jobs to complete...")

        # Keep looping until all jobs are in a terminal state
        while True:
            unfinished_jobs = [job for job in jobs if not job.done()]

            if not unfinished_jobs:  # all jobs are done
                break

            for job in unfinished_jobs:
                print(f"Current state for {job.display_name}: {job.state}")

            time.sleep(poll_interval)


if __name__ == "__main__":
    inference_manager = VertexBatchInferenceManager()

    inference_manager.run_batch_inference_on_base_model(
        DATASET_VERSION,
        sys_prompt_v=SYSTEM_PROMPT_VERSION,
        sets_to_run=["test"],
        wait=False,
    )
    inference_manager.run_batch_inference(
        DATASET_VERSION,
        sys_prompt_v=SYSTEM_PROMPT_VERSION,
        balanced_data=False,
        sets_to_run=["train", "val", "test"],
    )
