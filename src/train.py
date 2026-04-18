import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys 
from features.hidden_extractor import FeatureAccumulator, ExtractedFeatures, preprocess
from models.classifier import HallucinationClassifier

sys.path.insert(0, "src")

def extract_features(model, tokenizer, df, device, probe_layers=None):
    """извлечение признаков для всего датасета"""
    accumulator = FeatureAccumulator(model, probe_layers)
    hidden_size = model.config.hidden_size
    
    unc_list, int_list, probe_list, label_list = [], [], [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        token_ids, answer_start_idx = preprocess(tokenizer, row["prompt"], row["model_answer"])
        token_ids = token_ids.to(device)
        
        with accumulator:
            with torch.no_grad():
                outputs = model(token_ids)
        
        features = accumulator.compute_features(
            outputs.logits,
            token_ids,
            answer_start_idx,
            hidden_size
        )
        
        if features is not None:
            unc_list.append(features.uncertainty)
            int_list.append(features.internal_scalars)
            probe_list.append(features.probe_vec)
            label_list.append(row["is_hallucination"])
    
    return {
        "uncertainty_X": np.stack(unc_list).astype(np.float32),
        "internal_scalar_X": np.stack(int_list).astype(np.float32),
        "probe_last_prompt": np.stack(probe_list).astype(np.float32),
        "labels": np.array(label_list, dtype=np.int32),
    }

def main():

    MODEL_PATH = "./gigachat_model"
    TRAIN_DATA = "data/train.csv"
    OUTPUT_MODEL = "models/classifier.pkl"
    PROBE_LAYERS_STR = "0,5,10,15,20,25"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {MODEL_PATH} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()
  
    df_train = pd.read_csv(TRAIN_DATA)
    print(f"Train data size: {len(df_train)}")

    probe_layers = [int(x) for x in PROBE_LAYERS_STR.split(",")]
    features = extract_features(model, tokenizer, df_train, device, probe_layers)
    
  
    classifier = HallucinationClassifier(
        n_features=50,
        use_feature_selection=True,
        random_seed=42
    )
    classifier.fit(features)
    

    classifier.save(OUTPUT_MODEL)
    print(f"classifier saved to {OUTPUT_MODEL}")
    
if __name__ == "__main__":
    main()