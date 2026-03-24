import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import pandas as pd
from typing import Optional, List
from src.qwen_local_classification.config import Config
from src.qwen_local_classification.model import LLMClassifier

class ModelEvaluator:
    def __init__(self, config: Config, llm_classifier: LLMClassifier):
        self.config = config
        self.llm = llm_classifier

    def run_inference_loop(self, df: pd.DataFrame, batch_size: int = 16, boost_factors: Optional[List[float]] = None):
        device = self.llm.model.device
        df['token_length'] = df['text'].apply(lambda x: len(self.llm.tokenizer.encode(x, add_special_tokens=False)))
        # Sort by token length for efficient batching if needed, but for strictly matching indices back, avoid sort or re-sort.
        # Actually, for test predictions, we must keep the original order to match row-by-row accurately.
        # So we won't sort unless it's evaluation, or we sort and restore. Let's just track original index.
        df['orig_idx'] = df.index
        df_sorted = df.sort_values(by='token_length').reset_index(drop=True)
        
        results = []
        
        if boost_factors is not None:
            # boost factors should match number_token_ids length (in original script it included +1, here let's ensure it matches num classes)
            # The test inference script used torch.tensor([2.5, 1.0, 2.5, 1.0]) for 4 items 
            print(f"Using boost factors: {boost_factors}")
            boost_tensor = torch.tensor(boost_factors, device=device)
        else:
            boost_tensor = None

        with torch.inference_mode():
            for i in tqdm(range(0, len(df_sorted), batch_size), desc="Evaluating"):
                batch = df_sorted.iloc[i:i+batch_size]
                
                # Assume text is already formatted without the final label
                prompts = batch['text'].tolist()
                
                inputs = self.llm.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_seq_length).to(device)
                logits = self.llm.model(**inputs).logits
                last_idxs = inputs.attention_mask.sum(1) - 1
                last_logits = logits[torch.arange(len(batch)), last_idxs, :]
                
                last_logits_subset = last_logits[:, self.llm.number_token_ids]
                
                if boost_tensor is not None:
                    adjusted_logits = last_logits_subset + torch.log(boost_tensor)
                else:
                    adjusted_logits = last_logits_subset
                
                probs = F.softmax(adjusted_logits.float(), dim=-1)
                preds = torch.argmax(probs, dim=-1).cpu().numpy()
                
                for j in range(len(batch)):
                    res_dict = {
                        "orig_idx": batch['orig_idx'].iloc[j],
                        "text": batch['text'].iloc[j],
                        "pred": preds[j],
                        "probs": probs[j].cpu().numpy(),
                    }
                    if 'label' in batch.columns:
                        res_dict['true'] = batch['label'].iloc[j]
                    results.append(res_dict)

        # Restore original order
        results.sort(key=lambda x: x['orig_idx'])
        return results

    def evaluate(self, val_df: pd.DataFrame, batch_size: int = 16, display: int = 50):
        self.llm.load_finetuned_model(oversampled=True)
        results = self.run_inference_loop(val_df, batch_size)
        
        correct = sum([r['pred'] == r['true'] for r in results])
        accuracy = 100 * correct / len(val_df)
        
        print("\n--- Random samples ---")
        for s in random.sample(results, min(display, len(results))):
            print(f"\nText: {s['text'][:200]}...")
            print(f"True: {s['true']}  Pred: {s['pred']} {'✅' if s['pred'] == s['true'] else '❌'}")
            print("Probs:", ", ".join([f"{k}: {v:.3f}" for k, v in enumerate(s['probs'])]))
        
        print(f"\nValidation accuracy: {accuracy:.2f}% ({correct}/{len(val_df)})")
        print(f"Actual label distribution: \n{val_df['label'].value_counts()}")

    def predict(self, test_df: pd.DataFrame, batch_size: int = 16, boost_factors: Optional[List[float]] = None, output_file: str = "test_predictions.txt"):
        self.llm.load_finetuned_model(oversampled=True)
        results = self.run_inference_loop(test_df, batch_size, boost_factors)
        
        preds = [r['pred'] for r in results]
        assert len(preds) == len(test_df), "Length of test data and predictions does NOT match!"
        
        with open(output_file, "w") as f:
            for p in preds:
                f.write(f"{int(p)}\n")
        print(f"Predictions saved to {output_file}")
