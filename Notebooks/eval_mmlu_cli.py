import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

def get_log_likelihood(prompt: str, answer: str, model, tokenizer, device) -> float:
    full_input = prompt + answer
    input_ids = tokenizer(full_input, return_tensors="pt").input_ids.to(device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    target_ids = input_ids.clone()
    target_ids[:, :prompt_ids.shape[1]] = -100  # ignore prompt

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = target_ids[:, 1:].contiguous()

    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return -loss.sum().item()

def evaluate_subject(model, tokenizer, subject, num_samples, device):
    try:
        dataset = load_dataset("cais/mmlu", subject, split=f"test[:{num_samples}]")
    except Exception as e:
        print(f"Failed to load subject {subject}: {e}")
        return None

    correct = 0
    total = 0

    for sample in dataset:
        question = sample["question"]
        choices = sample["choices"]
        correct_answer = sample["answer"]

        prompt = f"Q: {question}\nChoices:\n"
        for idx, choice in enumerate(choices):
            prompt += f"{chr(65 + idx)}. {choice}\n"
        prompt += "Answer: "

        scores = [get_log_likelihood(prompt, choice, model, tokenizer, device) for choice in choices]
        pred = scores.index(max(scores))
        correct += int(pred == correct_answer)
        total += 1

    return (correct / total * 100) if total > 0 else None

def plot_scores(results, model_name):
    subjects = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(14, 6))
    plt.bar(subjects, scores)
    plt.xticks(rotation=90)
    plt.xlabel("Subject")
    plt.ylabel("Accuracy (%)")
    plt.title(f"GPT-2 Performance on MMLU Subjects")
    plt.tight_layout()
    plt.savefig(f"{model_name}_mmlu_scores.png")
    plt.show()

def main(model_name: str, num_samples: int):
    device = torch.device("cpu")
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    print("Fetching MMLU subjects...")
    subjects = get_dataset_config_names("cais/mmlu")

    results = {}
    for subject in tqdm(subjects, desc="Evaluating all MMLU subjects"):
        acc = evaluate_subject(model, tokenizer, subject, num_samples, device)
        if acc is not None:
            results[subject] = acc
            print(f"{subject}: {acc:.2f}%")

    plot_scores(results, model_name.replace("/", "_"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a HuggingFace LLM on all MMLU subjects.")
    parser.add_argument("--model", type=str, default="gpt2", help="Model ID from Hugging Face (e.g., gpt2)")
    parser.add_argument("--num_samples", type=int, default=25, help="Number of test samples per subject")
    args = parser.parse_args()

    main(args.model, args.num_samples)

