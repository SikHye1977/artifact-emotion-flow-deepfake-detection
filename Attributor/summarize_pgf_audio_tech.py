import re
from pathlib import Path

LOGS = {
    "Score-only MLP": "logs_compare/pgf_audio_tech_score.log",
    "Embedding MLP": "logs_compare/pgf_audio_tech_embedding.log",
    "AAT": "logs_compare/pgf_audio_tech_aat.log",
}

def extract_metric(text, name):
    pattern = rf"{re.escape(name)}:\s*([0-9.]+)"
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

print()
print("| Model | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Test Weighted-F1 |")
print("|---|---:|---:|---:|---:|---:|")

for name, log in LOGS.items():
    p = Path(log)
    if not p.exists():
        print(f"[WARN] missing: {log}")
        continue

    text = p.read_text(errors="ignore")

    val_acc = extract_metric(text, "Val  Acc")
    val_f1 = extract_metric(text, "Val  Macro-F1")
    test_acc = extract_metric(text, "Test Acc")
    test_f1 = extract_metric(text, "Test Macro-F1")
    test_wf1 = extract_metric(text, "Test Weighted-F1")

    print(
        f"| {name} | "
        f"{val_acc:.4f} | "
        f"{val_f1:.4f} | "
        f"{test_acc:.4f} | "
        f"{test_f1:.4f} | "
        f"{test_wf1:.4f} |"
    )

print()