import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.environ.get("PROJECT_ID")
if not PROJECT_ID:
    PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))

LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

BUCKET_NAME = os.environ.get("BUCKET_NAME", "your_bucket_name")
BUCKET_URI = f"gs://{BUCKET_NAME}"


DATASET_VERSION: str = "V15"
SYSTEM_PROMPT_VERSION: int = 3

BALANCED_DATA: bool = False

BASE_MODEL: str = "gemini-2.5-flash-lite"

N_EPOCHS: int = 8
ADAPTER_SIZE: str = "ADAPTER_SIZE_SIXTEEN"
LRM: int = 2
