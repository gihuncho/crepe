import torch
from transformers import AutoTokenizer
from .models import BertForCategoryErrorRegression

def load_model_and_tokenizer(
    model_name_or_path="gihuncho/crepe-biomedbert",
    cache_dir=None
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    model = BertForCategoryErrorRegression.from_pretrained(model_name_or_path, cache_dir=cache_dir)

    return model, tokenizer

def compute(
    model,
    tokenizer,
    reference_report,
    candidate_report,
    device="cpu",
):
    inputs = tokenizer(
        reference_report,
        candidate_report,
        return_tensors="pt",
        padding='max_length', # Pad to max_length
        truncation=True,
        max_length=512
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        if outputs.all_logits is None or len(outputs.all_logits) != model.num_error_categories:
            raise ValueError("Model did not return the expected count of logits.")
        predicted_error_counts_raw = [logit.item() for logit in outputs.all_logits]
    predicted_error_counts = [max(0, count) for count in predicted_error_counts_raw]

    return {
        "crepe_score": sum(predicted_error_counts),
        "predicted_error_counts": predicted_error_counts,
    }
