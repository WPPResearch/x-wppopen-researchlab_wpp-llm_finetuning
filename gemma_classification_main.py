from src.gemma_classification.config import (
    BALANCED_DATA,
    DATASET_VERSION,
    LR,
    N_EPOCHS,
    SYSTEM_PROMPT_VERSION,
)
from src.gemma_classification.vertex_dataset_processor import VertexDatasetProcessor
from src.gemma_classification.vertex_finetuning_manager import VertexTuningManager

if __name__ == "__main__":
    print(f"--- Starting Pipeline for {DATASET_VERSION} ---")

    processor = VertexDatasetProcessor(DATASET_VERSION, SYSTEM_PROMPT_VERSION)

    processor.check_local_data()
    processor.run_finetuning_pipeline()
    processor.run_testing_pipeline()

    print("All datasets processed successfully.")

    manager = VertexTuningManager()
    manager.sync_data_to_gcs(local_data_path="data/preprocessed_data", overwrite=False)
    job = manager.submit_custom_job(
        dataset_version=DATASET_VERSION,
        sys_prompt_v=SYSTEM_PROMPT_VERSION,
        balanced_data=BALANCED_DATA,
        epochs=N_EPOCHS,
        lr=LR,
    )
