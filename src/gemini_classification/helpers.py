import json
from pathlib import Path
from typing import List, Dict, Any, Union
from google.cloud import storage
from src.gemini_classification.config import BASE_MODEL


# --- FILE & GCS HELPERS ---


def verify_local_datasets(dataset_version: str) -> None:
    """Checks if Git LFS files exist in the expected directory."""
    target_dir = Path(f"data/raw_data/{dataset_version}")

    if not target_dir.exists():
        print(
            f"Error: Directory {target_dir} not found. Ensure Git LFS is initialized."
        )
        return

    files = list(target_dir.glob("*.jsonl"))
    if not files:
        print(f"Warning: No .jsonl files found in {target_dir}")
    else:
        print(f"Found {len(files)} local files for version {dataset_version}")


def save_jsonlines(filename: Union[str, Path], data_list: List[Dict[str, Any]]) -> None:
    """Saves a list of dictionaries to a file in JSONL format."""
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure dir exists

    with file_path.open("w", encoding="utf-8") as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Successfully saved {len(data_list)} rows to {file_path}")


def upload_directory_to_gcs(
    local_path: Path,
    bucket_name: str,
    project_id: str,
    location: str,
    overwrite: bool = False,
):
    """Uploads directory tree to GCS."""
    storage_client = storage.Client(project=project_id)
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except Exception:
        print(f"Creating bucket {bucket_name}...")
        bucket = storage_client.create_bucket(bucket_name, location=location)

    for file in local_path.rglob("*"):
        if file.is_file():
            remote_path = str(file.relative_to(local_path))
            blob = bucket.blob(remote_path)
            if not overwrite and blob.exists():
                continue
            blob.upload_from_filename(str(file), timeout=600)


# --- VERTEX AI STRING HELPERS ---


def generate_model_display_name(
    dataset_version: str,
    sys_prompt_v: int,
    balanced_data: bool,
    n_epochs: int,
    adapter_size: str,
    lrm: Union[int, float],
) -> str:
    version = dataset_version
    balanced = "_balanced" if balanced_data else ""
    return f"ft_{BASE_MODEL}_{version}_{adapter_size.lower()}_epochs_{n_epochs}_lr_{lrm}_sysprompt_{sys_prompt_v}{balanced}"


def format_gcs_inference_path(
    split: str, metadata: dict, sys_v: int, version: str
) -> str:
    spec = metadata["supervisedTuningSpec"]
    if split in ["train", "val"]:
        key = "trainingDatasetUri" if split == "train" else "validationDatasetUri"
        return spec[key].replace("for_finetuning_gemini", "for_evaluation_gemini")

    val_uri = spec["validationDatasetUri"]
    base_path = val_uri.split("/for_finetuning_gemini/")[0]
    return f"{base_path}/for_testing_gemini/{version}/test_sys_prompt_{sys_v}_{version}.jsonl"
