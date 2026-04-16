from unsloth import FastLanguageModel, tokenizer_utils
import torch
from typing import List, Union, Any, Dict
from transformers import DataCollatorForLanguageModeling
from src.qwen_local_classification.config import Config


# Prevent fixing untrained tokens automatically
def do_nothing(*args, **kwargs):
    pass


tokenizer_utils.fix_untrained_tokens = do_nothing


class DataCollatorForLastTokenLM(DataCollatorForLanguageModeling):
    def __init__(
        self,
        reverse_map: Dict[int, int],
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.ignore_index = ignore_index
        self.reverse_map = reverse_map

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        for i in range(len(examples)):
            last_token_idx = (
                (batch["labels"][i] != self.ignore_index).nonzero()[-1].item()
            )
            batch["labels"][i, :last_token_idx] = self.ignore_index
            batch["labels"][i, last_token_idx] = self.reverse_map[
                batch["labels"][i, last_token_idx].item()
            ]

        return batch


class LLMClassifier:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.number_token_ids = []
        self.reverse_map = {}

    def get_number_token_ids(self) -> List[int]:
        ids = []
        for i in range(0, self.config.num_classes):
            ids.append(self.tokenizer.encode(str(i), add_special_tokens=False)[0])
        return ids

    def load_base_model(self):
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            load_in_4bit=self.config.load_in_4bit,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
        )
        self.number_token_ids = self.get_number_token_ids()
        self.reverse_map = {
            value: idx for idx, value in enumerate(self.number_token_ids)
        }

    def trim_lm_head(self):
        """Trims lm_head to only output the specific number tokens classes."""
        par = torch.nn.Parameter(self.model.lm_head.weight[self.number_token_ids, :])
        self.model.lm_head.weight = par

    def setup_peft(self):
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.r,
            target_modules=[
                "lm_head",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.config.seed,
            use_rslora=self.config.use_rslora,
        )

    def load_finetuned_model(self, oversampled: bool = True):
        model_name = (
            self.config.finetuned_oversampled_model_name
            if oversampled
            else self.config.finetuned_model_name
        )
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            load_in_4bit=self.config.load_in_4bit,
            max_seq_length=self.config.max_seq_length,
            dtype=self.config.dtype,
        )
        FastLanguageModel.for_inference(self.model)
        # we need num classes + 1 in the existing test scripts or we can adjust logic to just use classes size
        self.number_token_ids = self.get_number_token_ids()  # 3 classes here

    def reconstruct_and_save(self, save_dir: str):
        """Restores full LM head shape before saving so huggingface format is valid."""
        FastLanguageModel.for_inference(self.model)

        old_size = self.model.config.vocab_size
        trimmed_lm_head = self.model.lm_head.weight.data.clone()
        trimmed_lm_head_bias = (
            self.model.lm_head.bias.data.clone()
            if hasattr(self.model.lm_head, "bias")
            and self.model.lm_head.bias is not None
            else torch.zeros(len(self.number_token_ids), device=trimmed_lm_head.device)
        )

        hidden_dim = trimmed_lm_head.shape[1]
        new_lm_head = torch.full(
            (old_size, hidden_dim),
            0,
            dtype=trimmed_lm_head.dtype,
            device=trimmed_lm_head.device,
        )
        new_lm_head_bias = torch.full(
            (old_size,),
            -1000.0,
            dtype=trimmed_lm_head_bias.dtype,
            device=trimmed_lm_head_bias.device,
        )

        for new_idx, orig_token_id in enumerate(self.number_token_ids):
            new_lm_head[orig_token_id] = trimmed_lm_head[new_idx]
            new_lm_head_bias[orig_token_id] = trimmed_lm_head_bias[new_idx]

        with torch.no_grad():
            new_lm_head_module = torch.nn.Linear(
                hidden_dim, old_size, bias=True, device=self.model.device
            )
            new_lm_head_module.weight.data.copy_(new_lm_head)
            new_lm_head_module.bias.data.copy_(new_lm_head_bias)
            self.model.lm_head.modules_to_save["default"] = new_lm_head_module

        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def get_data_collator(self):
        return DataCollatorForLastTokenLM(
            reverse_map=self.reverse_map, tokenizer=self.tokenizer
        )
