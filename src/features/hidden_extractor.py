import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ExtractedFeatures:
    
    uncertainty: np.ndarray      # 19 uncertainty features
    internal_scalars: np.ndarray # признаки со слоев
    probe_vec: np.ndarray        # hidden state последнего слоя

class FeatureAccumulator:
    
    
    def __init__(self, model, probe_layers: List[int] = None):
        self.model = model
        self.probe_layers = probe_layers or [0, 5, 10, 15, 20, 25]
        self._hooks = []
        self._hidden = {}
        
        
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            self.num_layers = len(model.model.layers)
        else:
            self.num_layers = len(model.layers)
            
     
        self.probe_layers = [l for l in self.probe_layers if l < self.num_layers]
        
    def attach(self):
        self._hidden.clear()
        for idx in self.probe_layers:
            name = f"layer_{idx}"
            
            def _make(n):
                def _fn(_, __, out):
                    h = out[0] if isinstance(out, tuple) else out
                    self._hidden[n] = h.detach()
                return _fn
            
            layer = self.model.model.layers[idx] if hasattr(self.model, 'model') else self.model.layers[idx]
            self._hooks.append(layer.register_forward_hook(_make(name)))
    
    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
    
    def __enter__(self):
        self.attach()
        return self
    
    def __exit__(self, *args):
        self.detach()
    
    def compute_features(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        answer_start: int,
        hidden_size: int
    ) -> Optional[ExtractedFeatures]:
        seq_len = input_ids.shape[1]
        n_answer = seq_len - answer_start
        
        if n_answer == 0:
            return None
            
        uncertainty = self._compute_uncertainty_features(
            logits, input_ids, answer_start, n_answer
        )
        internal_scalars = self._compute_internal_features(
            input_ids, answer_start, seq_len, n_answer, hidden_size
        )
        probe_vec = self._get_probe_vector(answer_start)
        
        self._hidden.clear()
        
        return ExtractedFeatures(
            uncertainty=uncertainty,
            internal_scalars=internal_scalars,
            probe_vec=probe_vec
        )
    
    def _compute_uncertainty_features(self, logits, input_ids, answer_start, n_answer):
        '''признаки неопределенности'''
        answer_logits = logits[0, answer_start - 1: input_ids.shape[1] - 1, :].float()
        answer_ids = input_ids[0, answer_start:input_ids.shape[1]]
        
        log_probs = torch.log_softmax(answer_logits, dim=-1)
        token_lp = log_probs.gather(1, answer_ids.unsqueeze(1)).squeeze(-1)
        
        probs = torch.softmax(answer_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        top1 = probs.max(dim=-1).values
        top5 = probs.topk(min(5, probs.shape[-1]), dim=-1).values.sum(dim=-1)
        
        probs_sorted, _ = probs.sort(dim=-1, descending=True)
        if probs_sorted.shape[-1] >= 2:
            top1_top2 = (probs_sorted[:, 0] - probs_sorted[:, 1]).mean().item()
        else:
            top1_top2 = 0.0
        
        
        lp_mean = token_lp.mean().item()
        lp_std = token_lp.std().item() if n_answer > 1 else 0.0
        cv = lp_std / (abs(lp_mean) + 1e-6)
        
        
        entropy_seq = entropy.cpu().numpy()
        if len(entropy_seq) > 1:
            slope = np.polyfit(range(len(entropy_seq)), entropy_seq, 1)[0]
            n_spikes = (entropy_seq > np.percentile(entropy_seq, 90)).sum()
        else:
            slope = 0.0
            n_spikes = 0
        
        return np.array([
            token_lp.mean().item(),
            token_lp.min().item(),
            token_lp.max().item(),
            lp_std,
            entropy.mean().item(),
            entropy.max().item(),
            entropy.std().item() if n_answer > 1 else 0.0,
            torch.exp(-token_lp.mean()).item(),
            float(n_answer),
            token_lp[0].item(),
            top1.mean().item(),
            top5.mean().item(),
            top1_top2,
            cv,
            slope,
            float(n_spikes),
            entropy_seq[0] if len(entropy_seq) > 0 else 0.0,
            entropy_seq[-1] if len(entropy_seq) > 0 else 0.0,
            (entropy_seq[-1] - entropy_seq[0]) if len(entropy_seq) > 1 else 0.0,
        ], dtype=np.float32)
    
    def _compute_internal_features(self, input_ids, answer_start, seq_len, n_answer, hidden_size):
        """внутренние признаки со слоев"""
        int_scalars = []
        
        for idx in self.probe_layers:
            hs = self._hidden[f"layer_{idx}"][0]
            int_scalars.append(hs[answer_start - 1].norm().item())
            int_scalars.append(hs[answer_start:seq_len].norm(dim=-1).mean().item())
            
            if n_answer > 1:
                int_scalars.append(hs[answer_start:seq_len].norm(dim=-1).std().item())
            else:
                int_scalars.append(0.0)
            
            # Logit Lens
            ans_hs = hs[answer_start - 1:seq_len - 1].unsqueeze(0)
            with torch.no_grad():
                if hasattr(self.model, 'lm_head'):
                    norm_layer = self.model.model.norm if hasattr(self.model, 'model') else self.model.norm
                    ll = self.model.lm_head(norm_layer(ans_hs)).float()
                else:
                    ll = ans_hs
                    
            ll_p = torch.softmax(ll[0], dim=-1)
            ll_e = -(ll_p * torch.log(ll_p + 1e-10)).sum(dim=-1)
            ll_max_prob = ll_p.max(dim=-1).values
            
            int_scalars.append(ll_e.mean().item())
            int_scalars.append(ll_e.std().item())
            int_scalars.append(ll_max_prob.mean().item())
        
        first_e = int_scalars[3]
        last_e = int_scalars[-3]
        int_scalars.append(first_e - last_e)
        int_scalars.append(last_e / (first_e + 1e-10))
        
        return np.array(int_scalars, dtype=np.float32)
    
    def _get_probe_vector(self, answer_start):
        """probe вектор с последнего слоя"""
        last_hs = self._hidden[f"layer_{self.probe_layers[-1]}"][0]
        return last_hs[answer_start - 1].cpu().float().numpy()
    

def preprocess(tokenizer, prompt: str, answer: str):

    messages_prompt = [{"role": "user", "content": prompt}]
    prompt_enc = tokenizer.apply_chat_template(
        messages_prompt,
        add_generation_prompt=True,
        tokenize=True,
        )
    prompt_token_ids = prompt_enc["input_ids"]
    answer_start_idx = len(prompt_token_ids)

    messages_full = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
        ]
    full_enc = tokenizer.apply_chat_template(messages_full, tokenize=True)
    token_ids = full_enc["input_ids"]
    token_ids = torch.tensor([token_ids], dtype=torch.long)

    return token_ids, answer_start_idx

def main():
   
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
   
    model_dir = "./gigachat_model"
    prompt = "Тестовый вопрос."
    answer = "Тестовый ответ."
    
   
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    
    input_ids, answer_start = preprocess(tokenizer, prompt, answer)
    input_ids = input_ids.to(next(model.parameters()).device)

    accumulator = FeatureAccumulator(model)
    with accumulator:
        with torch.no_grad():
            outputs = model(input_ids)
    
    features = accumulator.compute_features(
        outputs.logits, input_ids, answer_start, hidden_size=model.config.hidden_size
    )


if __name__ == "__main__":
    main()