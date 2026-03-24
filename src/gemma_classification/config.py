import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.environ.get("PROJECT_ID")
if not PROJECT_ID:
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

BUCKET_NAME = "ds_finetuning_for_campaign_outcomes"
BUCKET_URI = f"gs://{BUCKET_NAME}"


DATASET_VERSION: str = "V15"
SYSTEM_PROMPT_VERSION: int = 3

BALANCED_DATA: bool = False

BASE_MODEL: str = "google/gemma-3-1b-it"
HF_TOKEN = os.environ.get("HF_TOKEN")

N_EPOCHS: int = 8
LR: float = 2e-3
