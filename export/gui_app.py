"""Tkinter GUI for running fake news classification interactively.

This script loads the exported model once at startup and provides a simple
text area where users can paste an article, run inference, and decide whether
to analyze another sample or close the application. It also supports execution
from a PyInstaller one-file bundle by resolving the bundled data path via
``sys._MEIPASS``.
"""
from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, scrolledtext

from inference import load_model_and_tokenizer, predict


class FakeNewsApp:
    """GUI wrapper around the fake news classifier."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Fake News Detector")
        self.root.geometry("720x540")

        self.export_dir = resolve_export_dir()
        self.status_var = tk.StringVar(value="Loading model…")
        self.result_var = tk.StringVar(value="")

        self._build_layout()
        self._load_model()

    def _build_layout(self) -> None:
        """Lay out widgets for text entry, results, and actions."""
        heading = tk.Label(
            self.root,
            text="Fake News Detector",
            font=("Arial", 18, "bold"),
        )
        heading.pack(pady=(16, 8))

        prompt = tk.Label(
            self.root,
            text="Paste or type a news article below, then click Run Analysis.",
            font=("Arial", 11),
        )
        prompt.pack(pady=(0, 12))

        self.text_input = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            height=14,
            font=("Arial", 11),
        )
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=16)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=12)

        self.run_button = tk.Button(
            button_frame,
            text="Run Analysis",
            command=self.run_analysis,
            state=tk.DISABLED,
            width=14,
        )
        self.run_button.grid(row=0, column=0, padx=8)

        clear_button = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_text,
            width=10,
        )
        clear_button.grid(row=0, column=1, padx=8)

        quit_button = tk.Button(
            button_frame,
            text="Quit",
            command=self.root.destroy,
            width=10,
        )
        quit_button.grid(row=0, column=2, padx=8)

        self.result_label = tk.Label(
            self.root,
            textvariable=self.result_var,
            font=("Arial", 12, "bold"),
            fg="#1f6feb",
            wraplength=680,
            justify=tk.CENTER,
        )
        self.result_label.pack(pady=(4, 2))

        self.status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Arial", 10),
            fg="#555555",
            wraplength=680,
            justify=tk.CENTER,
        )
        self.status_label.pack(pady=(0, 12))

    def _load_model(self) -> None:
        """Load the model once and enable the Run button."""
        try:
            self.model, self.tokenizer, self.device = load_model_and_tokenizer(self.export_dir)
        except FileNotFoundError as exc:
            messagebox.showerror("Model files missing", str(exc))
            self.root.destroy()
            return
        except Exception as exc:  # pragma: no cover - defensive guard for runtime issues
            messagebox.showerror("Error loading model", str(exc))
            self.root.destroy()
            return

        self.status_var.set("Model loaded. Enter article text and click Run Analysis.")
        self.run_button.configure(state=tk.NORMAL)

    def run_analysis(self) -> None:
        """Run inference on the provided text and display the result."""
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Input required", "Please enter an article before running analysis.")
            return

        self.status_var.set("Running analysis…")
        self.result_var.set("")
        self.root.update_idletasks()

        try:
            label, probability = predict(text, self.model, self.tokenizer, self.device)
        except Exception as exc:  # pragma: no cover - displayed to user instead of crashing
            messagebox.showerror("Prediction error", str(exc))
            self.status_var.set("An error occurred while running inference.")
            return

        self.result_var.set(f"Prediction: {label} ({probability:.4f})")
        self.status_var.set("Done. You can analyze another article or close the app.")

    def clear_text(self) -> None:
        """Reset the text area and message labels."""
        self.text_input.delete("1.0", tk.END)
        self.result_var.set("")
        self.status_var.set("Enter article text and click Run Analysis.")


def resolve_export_dir() -> Path:
    """Return the path to the exported model directory, supporting PyInstaller."""
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path / "fake_news_model"


def main() -> None:
    root = tk.Tk()
    FakeNewsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()