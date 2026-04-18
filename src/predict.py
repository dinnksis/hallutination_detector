import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import time

from features.hidden_extractor import FeatureAccumulator, preprocess
from models.classifier import HallucinationClassifier

@dataclass
class ScoringResult:
    is_hallucination: bool
    is_hallucination_proba: float
    t_model_sec: float = 0.0
    t_overhead_sec: float = 0.0

class GuardianOfTruth:
    def __init__(self, model, tokenizer, classifier_path: str, probe_layers=None):
        self.model = model
        self.tokenizer = tokenizer
        self.classifier = HallucinationClassifier.load(classifier_path)
        self.accumulator = FeatureAccumulator(model, probe_layers)
        self.hidden_size = model.config.hidden_size
    
   
    def score(self, prompt: str, answer: str) -> ScoringResult:
        
        token_ids, answer_start_idx = preprocess(self.tokenizer, prompt, answer)
        device = next(self.model.parameters()).device
        token_ids = token_ids.to(device)
        
        t0 = time.perf_counter()
        with self.accumulator:
            with torch.no_grad():
                outputs = self.model(token_ids)
        t_model = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        features = self.accumulator.compute_features(
            outputs.logits,
            token_ids,
            answer_start_idx,
            self.hidden_size
        )
        del outputs   
        t_overhead = time.perf_counter() - t1
        
        if features is None:
            return ScoringResult(is_hallucination=False, is_hallucination_proba=0.0)
        
        
        is_hall, prob = self.classifier.predict({
            "probe_vec": features.probe_vec,
            "internal_scalars": features.internal_scalars,
            "uncertainty": features.uncertainty,
        })
        
        return ScoringResult(
            is_hallucination=bool(is_hall),
            is_hallucination_proba=prob,
            t_model_sec=t_model,
            t_overhead_sec=t_overhead,
        )

    
def main():
    MODEL_PATH = "./gigachat_model"
    CLASSIFIER_PATH = "models/classifier.pkl"
    INPUT_CSV = "data/bench/knowledge_bench_private_no_labels.csv"
    OUTPUT_CSV = "data/bench/knowledge_bench_private_scores.csv"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
    
    guardian = GuardianOfTruth(model, tokenizer, CLASSIFIER_PATH)
    
    df = pd.read_csv(INPUT_CSV)
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        res = guardian.score(row["prompt"], row["model_answer"])
        results.append(res)
        
    df["pred_proba"] = [r.is_hallucination_proba for r in results]
    df["t_model_ms"] = [r.t_model_sec * 1000 for r in results]
    df["t_overhead_ms"] = [r.t_overhead_sec * 1000 for r in results]
    
    df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()