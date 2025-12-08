"""
Standalone inference script for the fake news classifier.

Usage:
    echo "Some article text" | python export/inference.py
or run interactively:
    python export/inference.py
    > Enter article text: ...

The script loads the exported tokenizer and model from the
`export/fake_news_model` directory relative to this file, runs a forward
pass, applies softmax, and prints the predicted label with its
probability.
"""
from pathlib import Path
import sys
from typing import Iterable, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = ["fake", "real"]


def _check_required_files(export_dir: Path, required_files: Iterable[str], weight_options: Iterable[str]) -> None:
    """Ensure the exported model directory contains the expected files."""

    missing = [filename for filename in required_files if not (export_dir / filename).exists()]
    has_weight = any((export_dir / filename).exists() for filename in weight_options)
    if missing or not has_weight:
        missing_items = missing.copy()
        if not has_weight:
            missing_items.append(f"one of: {', '.join(weight_options)}")
        missing_list = ", ".join(missing_items)
        raise FileNotFoundError(
            f"Missing required files in {export_dir}: {missing_list}. "
            "Ensure the exported tokenizer and model weights (e.g., model.safetensors) "
            "are placed in this directory before running inference."
        )


def load_model_and_tokenizer(export_dir: Path) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, torch.device]:
    """Load the tokenizer and model from the export directory and pick a device."""
    _check_required_files(
        export_dir,
        required_files=(
            "config.json",
            "tokenizer.json",
        ),
        weight_options=(
            "model.safetensors",
        ),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(export_dir)
    model = AutoModelForSequenceClassification.from_pretrained(export_dir)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def read_input_text() -> str:
    """Read article text from stdin (piped) or interactively."""
    if not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        text = input("Enter article text: ")
    return text.strip()


def predict(text: str, model: AutoModelForSequenceClassification, tokenizer: AutoTokenizer, device: torch.device) -> Tuple[str, float]:
    """Run the classifier and return the predicted label and probability."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits.squeeze(0)
        probabilities = torch.softmax(logits, dim=-1)
    prob_value, idx = torch.max(probabilities, dim=-1)
    label = LABELS[idx.item()]
    return label, prob_value.item()


def main() -> None:
    export_dir = Path(__file__).resolve().parent / "fake_news_model"
    text = read_input_text()
    if not text:
        print("No article text provided. Please pipe text to stdin or enter it interactively.")
        sys.exit(1)

    try:
        model, tokenizer, device = load_model_and_tokenizer(export_dir)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    label, probability = predict(text, model, tokenizer, device)
    print(f"Prediction: {label} ({probability:.4f})")


if __name__ == "__main__":
    main()