"""Compare Full Fine-Tuning, LoRA, QLoRA and Prefix-Tuning on Qwen2-0.5B-Instruct
for strategy-conditioned counter-speech generation (IntentCONANv2).

Usage:
    pip install -r requirements.txt
    export HF_TOKEN=...   # optional, only needed if the Hub rate-limits you
    python src/peft_comparison.py

Writes results/metrics.csv. Needs a CUDA GPU (developed on a single Tesla T4).
"""

import gc
import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

# ---------------------------------------------------------------- config
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
DATASET_NAME = "Aswini123/IntentCONANv2"
MAX_SEQ_LENGTH = 2048
NUM_TRAIN_EPOCHS = 1
MAX_STEPS = -1  # set e.g. 100 for a quick smoke test; overrides epochs when > 0
LEARNING_RATE = 5e-5
SEED = 42
GEN_BATCH_SIZE = 4
MAX_NEW_TOKENS = 128

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

INFERENCE_TEMPLATE = """Below is a hate speech message and a required counter-speech strategy. Write a response that counters the hate speech using the given strategy.

### Hate Speech:
{hatespeech}

### Strategy:
{csType}

### Counter-speech Response:
"""
PROMPT_TEMPLATE = INFERENCE_TEMPLATE + "{counterspeech}"
RESPONSE_MARKER = "### Counter-speech Response:"


# ---------------------------------------------------------------- helpers
def print_gpu_memory(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage}] GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_gpu_memory("After cleanup")


def tokenize_data(examples, tokenizer):
    texts = [
        PROMPT_TEMPLATE.format(hatespeech=hs, csType=cs_type, counterspeech=cs) + tokenizer.eos_token
        for hs, cs_type, cs in zip(examples["hatespeech"], examples["csType"], examples["counterspeech"])
    ]
    tokenized = tokenizer(texts, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)
    tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
    return tokenized


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"  # required for batched generation with decoder-only models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def lora_config():
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def build_model(method, use_bf16):
    """Return the model for one method, with the right PEFT wrapper applied."""
    if method == "QLoRA":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map={"": 0})
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config())
    else:
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, device_map="auto")
        if method == "LoRA":
            model = get_peft_model(model, lora_config())
        elif method == "Prefix-Tuning":
            model = get_peft_model(
                model,
                PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20, prefix_projection=True),
            )

    if method == "Full Fine-Tuning":
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,} (100%)")
    else:
        model.print_trainable_parameters()
    return model


def train(model, tokenizer, train_dataset, method, use_bf16):
    full_ft = method == "Full Fine-Tuning"
    formatted_train = train_dataset.map(
        lambda x: tokenize_data(x, tokenizer), batched=True, remove_columns=train_dataset.column_names
    )
    # Effective batch size is 16 for every method; full FT trades batch for accumulation to fit a T4.
    args = TrainingArguments(
        output_dir=f"outputs/{method.lower().replace(' ', '_')}",
        per_device_train_batch_size=1 if full_ft else 2,
        gradient_accumulation_steps=16 if full_ft else 8,
        gradient_checkpointing=full_ft,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        bf16=use_bf16,
        fp16=False,
        logging_steps=10,
        save_strategy="no",
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=10,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
    )
    model.train()
    Trainer(
        model=model,
        args=args,
        train_dataset=formatted_train,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True),
    ).train()
    print_gpu_memory("After training")


def generate(model, tokenizer, prompts):
    model.eval()
    responses = []
    for i in tqdm(range(0, len(prompts), GEN_BATCH_SIZE), desc="Generating"):
        batch = prompts[i : i + GEN_BATCH_SIZE]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # greedy, so metrics are deterministic
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for prompt, output in zip(batch, outputs):
            text = tokenizer.decode(output, skip_special_tokens=True)
            responses.append(
                text.split(RESPONSE_MARKER)[-1].strip() if RESPONSE_MARKER in text else text[len(prompt) :].strip()
            )
    return responses


# ---------------------------------------------------------------- metrics
def calculate_metrics(predictions, references, train_texts):
    import evaluate
    import nltk
    from bert_score import score as bert_score_fn
    from detoxify import Detoxify
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from textstat import flesch_reading_ease

    for pkg in ("punkt", "wordnet", "punkt_tab"):
        nltk.download(pkg, quiet=True)

    refs_nested = [[r] for r in references]
    results = {}
    results["BLEU"] = evaluate.load("bleu").compute(predictions=predictions, references=refs_nested)["bleu"]
    rouge = evaluate.load("rouge").compute(predictions=predictions, references=references)
    results["ROUGE-1"], results["ROUGE-2"], results["ROUGE-L"] = rouge["rouge1"], rouge["rouge2"], rouge["rougeL"]
    results["METEOR"] = evaluate.load("meteor").compute(predictions=predictions, references=references)["meteor"]
    results["GLEU"] = evaluate.load("google_bleu").compute(predictions=predictions, references=refs_nested)["google_bleu"]

    def repetition_rate(text):
        words = text.lower().split()
        return 1 - len(set(words)) / len(words) if words else 0

    def safe_flesch(text):
        try:
            return flesch_reading_ease(text)
        except Exception:
            return 0

    results["Repetition Rate"] = np.mean([repetition_rate(p) for p in predictions])
    results["Flesch Reading Ease"] = np.mean([safe_flesch(p) for p in predictions])

    # CoSIM: TF-IDF cosine between each prediction and its reference
    tfidf = TfidfVectorizer().fit_transform(predictions + references)
    n = len(predictions)
    results["CoSIM"] = np.mean([cosine_similarity(tfidf[i], tfidf[n + i])[0][0] for i in range(n)])

    _, _, f1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    results["BERT Score"] = f1.mean().item()

    results["Toxicity"] = np.mean(Detoxify("original").predict(predictions)["toxicity"])

    # Novelty: 1 - max TF-IDF similarity to any training target (500-sample subset for speed)
    train_sample = train_texts[:500]
    tfidf = TfidfVectorizer().fit_transform(predictions + train_sample)
    sims = cosine_similarity(tfidf[:n], tfidf[n:])
    results["Novelty"] = np.mean(1 - sims.max(axis=1))

    # Diversity: 1 - mean pairwise similarity between predictions
    sim = cosine_similarity(TfidfVectorizer().fit_transform(predictions))
    results["Diversity"] = 1 - np.mean(sim[np.triu_indices(n, k=1)])
    return results


# ---------------------------------------------------------------- main
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import login

        login(token=os.environ["HF_TOKEN"])

    split = load_dataset(DATASET_NAME)["train"].train_test_split(test_size=0.1, seed=SEED)
    train_dataset, test_dataset = split["train"], split["test"]
    print(f"Train: {len(train_dataset)}  Test: {len(test_dataset)}")

    test_prompts = [INFERENCE_TEMPLATE.format(hatespeech=ex["hatespeech"], csType=ex["csType"]) for ex in test_dataset]
    references = [ex["counterspeech"] for ex in test_dataset]
    train_targets = [ex["counterspeech"] for ex in train_dataset]

    use_bf16 = torch.cuda.is_bf16_supported()
    generations = {}
    for method in ["Full Fine-Tuning", "LoRA", "QLoRA", "Prefix-Tuning"]:
        print(f"\n{'=' * 60}\n{method}\n{'=' * 60}")
        tokenizer = load_tokenizer()
        model = build_model(method, use_bf16)
        train(model, tokenizer, train_dataset, method, use_bf16)
        generations[method] = generate(model, tokenizer, test_prompts)
        print(f"Sample: {generations[method][0]}\nReference: {references[0]}")
        del model, tokenizer
        cleanup_memory()

    results = pd.DataFrame(
        {m: calculate_metrics(preds, references, train_targets) for m, preds in generations.items()}
    ).T
    print(results.to_string())
    os.makedirs("results", exist_ok=True)
    results.to_csv("results/metrics.csv", index_label="Model")


if __name__ == "__main__":
    main()
