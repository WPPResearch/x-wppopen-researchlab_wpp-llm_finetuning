# AI Lab LLM Campaign Finetuning

*Requires Python: >= 3.10*

This repository contains the codebase for fine-tuning Large Language Models (LLMs) for classification campaigns. It supports multiple strategies depending on the model and environment you are using: from fully managed services on Google Cloud using Vertex AI for Gemini and Gemma, to running a custom local pipeline on a Conda-configured Virtual Machine.

## Project Structure

The project code is primarily organized into three separate workflows:

1. **Gemini Vertex AI Workflow (`gemini_classification_main.py`)** 
   - A fully managed pipeline to prepare data, finetune Google's Gemini models, and perform batch inference using Vertex AI.
2. **Gemma Vertex AI Custom Jobs Workflow (`gemma_classification_main.py`)**
   - An orchestration script to submit a custom training job to Vertex AI for fine-tuning open-weights models (principally Gemma).
3. **Local LLM Classifier (`src/qwen_local_classification/`)**
   - A standalone module for finetuning Qwen3 designed to run on a local GPU-equipped compute instance (e.g., via a Google Cloud Compute Engine VM using Conda). 

## Models Used

Depending on the workflow and configuration chosen, this project defaults to fine-tuning the following base models:
- **Gemini:** [Gemini 2.5 Flash Lite](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models) (via Vertex AI Model Garden)
- **Gemma:** [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) (requires Hugging Face token)
## Environment Setup (uv)

This project manages its python environment and dependencies using `uv` (a fast Python package installer and resolver). For the Gemini and Gemma workflows, you should utilize `uv` to abstract environment setup.

1. **Install uv:** Please install `uv` according to the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).
2. **Initialize Git LFS:** This repository uses Git LFS for raw `.jsonl` datasets. Ensure it is installed on your system, then run:
```bash
git lfs install
git lfs pull
```
This should download all the raw data files to your local machine in `data/raw_data/<DATASET_VERSION>` of the repository root directory.
3. **Sync Dependencies:** Ensure your local python version aligns (the project typically expects python 3.10+) and install dependencies from the lockfile:
   ```bash
   uv sync
   ```
   *This commands automatically discovers the `pyproject.toml` and creates a `.venv` directory for you locally.*

4. **Environment Variables:**
   This project loads configuration keys via `python-dotenv`. You must manually create a file named `.env` in the root of the project directory.

   - **For GCP services (Vertex AI, GCS):** Authenticate directly via the terminal (see instructions below). Do not store GCP Service Account JSONs here. Instead, define your working project ID and bucket name:
     ```env
     PROJECT_ID="your_google_cloud_project_id"
     BUCKET_NAME="your_gcs_bucket_name"
     ```
   - **For accessing gating Hugging Face Models (Gemma/Qwen):**
     ```env
     HF_TOKEN="your_hugging_face_token_here"
     ```

## Data Processing Pipeline

This project handles the cleaning and formatting of datasets already present in the repository (managed via Git LFS) before sending them for fine-tuning via `VertexDatasetProcessor` scripts. When you execute either the Gemini or Gemma main scripts, the preprocessing pipeline behaves as follows:

1. **Local Data Verification:** The script verifies that the required raw `.jsonl` files (e.g., `V15_train.jsonl`) exist within `data/raw_data/<DATASET_VERSION>`. Users must ensure they have run `git lfs pull` to swap Git LFS pointers for actual data.
2. **Formatter:** It cleans the JSON objects (formatting fields like `platform`, `audience`, etc.), balances target classes if configured, and wraps the text in designated system/user templates for conversational training. The formatting schemas differentiate slightly whether training Gemini (Vertex message objects) or Gemma (Hugging Face dialogue prompts).
3. **Splitter:** The formatter partitions the data, generating local JSONL outputs stored under `data/preprocessed_data/for_finetuning...`, `for_evaluation...`, and `for_testing...`. These splits are what eventually get synced up and uploaded to the GCS bucket dictated by the respective `config` files.

---

## 1. Fine-tuning Gemini Models (Vertex AI)

The main entry point for this workflow is `gemini_classification_main.py`. This script relies on configurations set in `src/gemini_classification/config.py`.

### How to use:

1. Make sure your environment is authenticated with Google Cloud Platform and has necessary permissions for Vertex AI. You should set up Application Default Credentials and login so the scripts can fetch your identity token:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project <PROJECT_ID>
   ```
2. Ensure you have the `uv` package manager or `pip`-installed requirements.
3. Configure the parameters in `src/gemini_classification/config.py` (e.g., `DATASET_VERSION`, `SYSTEM_PROMPT_VERSION`, hyperparameters).
4. Run the main script:
   ```bash
   uv run python gemini_classification_main.py
   ```
   **What the script does:**
   - Processes datasets and formats prompts.
   - Syncs data to Google Cloud Storage (GCS).
   - Submits a Supervised Fine-Tuning (SFT) job to Vertex AI to train a Gemini model.
   - Runs batch inference to evaluate the model on the validation and test splits.
   - Calculates and saves the metrics.

### Technical Finetuning Details:
- **Manner of Finetuning:** Executed via fully managed Vertex AI Supervised Tuning (`SupervisedTuningJob`). It trains a LoRA adapter against the frozen baseline Gemini weights.
- **Parameters:** Main variables exported include Epochs, Learning Rate Multiplier (LRM), and LoRA Adapter Size (e.g. `ADAPTER_SIZE_SIXTEEN`). 
- **Output Treatment:** Since Gemini is a generative Causal LM, classification outputs are treated as generative text. However, during the Batch Prediction job, we enforce rigorous schema adherence using Vertex AI's structured outputs (`"response_mime_type": "text/x.enum"` and `"response_schema": {"enum": ["0", "1", "2"]}`). This guarantees the generated string will precisely map back to one of the target taxonomy classes.

### Viewing Results:
After the pipeline completes `evaluate_results()`, you can find the performance outputs saved in the local `results/` folder:
- **Metrics Logs:** `results/classification/gemini_eval_summary.csv` (Contains F1 Macro, F1 Weighted, exact match summaries, etc.)
- **Confusion Matrices:** `results/classification/confusion_m/` (PNG plots visualising predicted versus true class distributions)

**Note on Raw Predictions:** The actual Batch Prediction output arrays (raw JSONL files containing the model's generated text for the validation and test sets) are saved directly in your Google Cloud Storage bucket. 
Vertex AI automatically creates a directory based on the prediction job name. You can expect to find the raw prediction dumps roughly under:
- **Base model evaluation (Control):** `gs://<BUCKET_NAME>/batch_predictions/<split>_set_pred_baseline_<model_name>_<version>/`
- **Fine-tuned model evaluation:** `gs://<BUCKET_NAME>/batch_predictions/<split>_set_pred_ft_<model_name>_<version>_<hyperparams>/`

---

## 2. Fine-tuning Gemma Models (Vertex AI Custom Training)

The entry point for this workflow is `gemma_classification_main.py`. It submits a custom containerized Python job to Vertex AI. It relies on configurations set in `src/gemma_classification/config.py`.

**Hardware Requirements:** The training script is optimized to run on Vertex instances provisioned at a minimum with an `NVIDIA_L4` GPU (such as `g2-standard-8` which provides 1xL4 GPU). This matches the requirements defined in the `vertex_finetuning_manager`. For larger versions of Gemma, you may need to increase the GPU memory and use the A100 machine type.

### How to use:

1. Ensure your environment is authenticated with Google Cloud Platform and has necessary permissions. Make sure to authenticate Docker to use Container Registry (GCR) or Artifact Registry:
   ```bash
   gcloud auth login
   gcloud config set project <PROJECT_ID>
   # Authenticate docker for gcr.io
   gcloud auth configure-docker gcr.io
   ```
2. Review and configure the parameters within `src/gemma_classification/config.py` (e.g., hyperparameters, base model paths). You must also ensure you export your Hugging Face access token so the script can download the gated Gemma models. [See how to create a token here](https://huggingface.co/docs/hub/security-tokens).
   ```bash
   export HF_TOKEN="your_hugging_face_token_here"
   ```
3. Ensure Docker is installed and running on your machine. Build and push the custom Docker container that the Vertex AI worker will use. Remember to replace `<PROJECT_ID>` with your actual Google Cloud Project ID:
   ```bash
   # From the root of the repository
   docker build -t gemma-classifier-test:latest -f src/gemma_classification/Dockerfile .
   docker tag gemma-classifier-test:latest gcr.io/<PROJECT_ID>/gemma-classifier-test:latest
   docker push gcr.io/<PROJECT_ID>/gemma-classifier-test:latest
   ```
4. Execute the pipeline:
   ```bash
   uv run python gemma_classification_main.py
   ```
   **What the script does:**
   - Processes and prepares datasets.
   - Uploads datasets to GCS.
   - Submits a custom Python training job using your provided worker scripts (located throughout `src/gemma_classification/finetune/`).
   - The remote custom training job runs the `train_gemma_sft.py` script.

### Technical Finetuning Details:
- **Manner of Finetuning:** Standard Supervised Fine-Tuning (SFT) using Hugging Face's `TRL (SFTTrainer)` on Vertex AI compute. The base model is quantized to `bfloat16`. 
- **Parameters:** Configured using the `transformers` TrainingArguments (Batch size, Gradient Accumulation, Cosine decay learning rate schedules). Additionally, we can specify `--freeze-bottom-layers` to only fine-tune the upper heads of the network.
- **Output Treatment:** During inference, Gemma behaves auto-regressively via `model.generate()`. However, we utilize a custom `LogitsProcessor` (`LabelConstraint`) to forcefully set all un-allowed token probabilities to negative infinity. This ensures the output generation strictly adheres to the one-token class limits (i.e. '0', '1', '2').
- **Important Note:** Because this highly constrained structure limits output generation to a single token, it is much more concise and demonstrably performs better in tests. Thus, this Gemma pipeline strictly only supports system prompts that encode classification labels as integers (ie `SYSTEM_PROMPT[3]`)!

### Viewing Results:
Since Gemma custom training completely executes on remote worker instances within Vertex AI, the model checkpoints are uploaded directly to your Google Cloud Storage bucket upon completion.

- **Model Checkpoints:** Saved under `gs://<BUCKET_NAME>/gemma_checkpoints/<job_display_name>/final_model/`.
- **Test Set Predictions:** The model generates inference on the test split right after training. The predictions file is stored in GCS at `gs://<BUCKET_NAME>/gemma_checkpoints/<job_display_name>/test_predictions.txt` and is automatically downloaded locally to `results/classification/submissions/test_set_pred_<job_display_name>.txt`.

*(Where `<job_display_name>` is a dynamically generated string from your hyperparameters, e.g. `ft_gemma-3-1b-it_V15_ep8_lr0.0002_sys3`)*

---

## 3. Local Open-Weights Classifier (Qwen3)

If you wish to manage the hardware directly and bypass Vertex AI's managed SFT services, use the pipeline in the `src/qwen_local_classification/` directory.

**Model used:** [unsloth/Qwen3-4B-Base](https://huggingface.co/unsloth/Qwen3-4B-Base) ([Unsloth](https://github.com/unslothai/unsloth)-optimized weights for efficient local fine-tuning)

### Environment & Hardware Setup

This code was designed to be run in a Conda environment on a Google Cloud Virtual Machine with the following configuration:

- **Location:** `europe-west1-b` or other
- **Machine Type:** `n1-highmem-4` (4 vCPUs, 26 GB Memory)
- **GPUs:** 1 x NVIDIA T4
- **Boot Disk Image:** `c0-deeplearning-common-cu123-v20250327-debian-11-py310-conda` (Debian 11, Python 3.10, CUDA 12.3)

### Installation

To set up the environment and install necessary dependencies using `uv`:

1. First, make sure you deactivate any other active environment (like Conda):
   ```bash
   conda deactivate
   ```

2. Sync the project with the `qwen` dependency group from the project root:
   ```bash
   uv sync --extra qwen
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

**Hugging Face Authentication**
Since `Unsloth` and the `transformers` library download pretrained models directly from Hugging Face, you need to ensure your environment is authenticated (especially for gated models).
Before running the training script, log in via your terminal:
```bash
huggingface-cli login
```
Provide your Hugging Face access token when prompted.

### How to use:

The main entry point for the pipeline is `qwen_classification_main.py`. You can execute different stages of the pipeline using the provided subcommands from the project root.

1. **Preprocess Data:** To preprocess the dataset and split it into training, validation, and testing sets:
   ```bash
   uv run python qwen_classification_main.py preprocess
   ```

2. **Train Model:** To finetune the model on the preprocessed training data:
   ```bash
   uv run python qwen_classification_main.py train
   ```
   *Note: Ensure your environment has the required Hugging Face authentication if the model weights are restricted.*

3. **Evaluate Model:** To evaluate the model on the validation dataset:
   ```bash
   uv run python qwen_classification_main.py evaluate
   ```

4. **Predict:** To generate predictions on the test dataset:
   ```bash
   uv run python qwen_classification_main.py predict
   ```

**The `--boost` Flag**
You can optionally apply confidence boosting adjustments to the model's intermediate logit values during prediction by appending the `--boost` flag:
```bash
uv run python qwen_classification_main.py predict --boost
```
Because LLM logit distributions often skew towards majority classes in imbalanced datasets (e.g., classifying "Average performance" far more often than edge-cases), passing the `--boost` flag manually multiplies the raw prediction logits for specific class numbers by static weighting factors **before** the final `softmax` mapping. 
Currently configured to `[2.5, 1.0, 2.5]`, this artificially boosts the likelihood of the `0` (Underperforming) and `2` (Overperforming) classes over the `1` (Average) class to recover minority-class sensitivity. This will shift the final predicted distribution when saving `test_predictions_oversampled_boosted.txt`.

### Technical Finetuning Details:
- **Manner of Finetuning:** Parameter-Efficient Fine-Tuning (PEFT) mapping LoRA matrices using the highly optimized `Unsloth` framework. By default, it updates the `lm_head`, and attention/projection modules (`q_proj`, `k_proj`, etc.).
- **Parameters:** Main variables exported in `config.py` dictate standard LoRA settings (`r`, `lora_alpha`, `use_rslora`, `lora_dropout`) along with optimizer configurations like 8-bit AdamW. Target precision handles `load_in_4bit` options conditionally.
- **Output Treatment:** During parameter initialization, the model's `lm_head` is forcibly _trimmed_ down to solely reflect the logits of the specific classification tokens (the string representation of integers mapped). Thus, instead of doing sequence-level auto-regressive text mapping, the architecture returns concrete probability distributions explicitly for the desired output classes.

### Configuration

Model parameters, file paths, and dataset versions are primarily managed in `src/qwen_local_classification/config.py`. Update this file to change the base model, generation parameters, or dataset pointers.

### Directory Structure & Data Paths

The pipeline expects explicit inputs and writes explicit outputs based on the `dataset_version` configured in `src/qwen_local_classification/config.py`.
Assuming the configured `dataset_version` is `V15`, the files interact with the following locations from the project root. (Substitute `V15` with your actual version variable such as `V1`, `V2`, etc.):

- **Inputs (Raw Data):** Preprocessing script expects `.jsonl` files in `data/raw_data/{dataset_version}/` (e.g., `{dataset_version}_train.jsonl` and `{dataset_version}_test.jsonl`).
- **Intermediate Outputs (Preprocessed Data):** Running `preprocess` writes CSV variants to `data/preprocessed_data/for_training_qwen/{dataset_version}/`, `for_evaluation_qwen/{dataset_version}/`, and `for_testing_qwen/{dataset_version}/`.
- **Artifacts (Predictions & Checkpoints):**
  - **Intermediate Checkpoints:** Saved to `qwen_finetuned/checkpoints/` during training.
  - **Adapter Weights:** Saved as `qwen_finetuned/finetuned-unsloth-oversampled-{dataset_version}` (or `finetuned-unsloth-{dataset_version}`).
  - **Predictions:** Inference evaluations exported to `results/classification/submissions/` with filenames like `QWEN_V15_lr0.0001_bs32_ep1_r16_3407.txt`.
