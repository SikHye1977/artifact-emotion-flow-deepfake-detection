import re
from pathlib import Path

LOGS = {
    "AAT small d64 L1": "logs_aat_tuning/aat_small.log",
    "AAT base d128 L1 drop0.4": "logs_aat_tuning/aat_base_dropout04.log",
    "AAT d128 L2 drop0.4": "logs_aat_tuning/aat_2layer_dropout04.log",
    "AAT wide d256 L1 drop0.4": "logs_aat_tuning/aat_wide_dropout04.log",
}

def extract_metric(text, name):
    pattern = rf"{re.escape(name)}:\s+([0-9.]+)"
    m = re.search(pattern, text)
    if not m:
        return None
    return float(m.group(1))

def parse_log(path):
    text = Path(path).read_text(errors="ignore")
    return {
        "Val Binary F1": extract_metric(text, "Val  binary F1"),
        "Val Modality F1": extract_metric(text, "Val  modality F1"),
        "Val Tech F1": extract_metric(text, "Val  tech F1"),
        "Test Binary F1": extract_metric(text, "Test binary F1"),
        "Test Modality F1": extract_metric(text, "Test modality F1"),
        "Test Tech F1": extract_metric(text, "Test tech F1"),
    }

rows = []
for model_name, log_path in LOGS.items():
    path = Path(log_path)
    if not path.exists():
        print(f"[WARN] missing log: {log_path}")
        continue
    rows.append((model_name, parse_log(path)))

headers = [
    "Model",
    "Val Binary",
    "Val Modality",
    "Val Tech",
    "Test Binary",
    "Test Modality",
    "Test Tech",
]

print()
print("| " + " | ".join(headers) + " |")
print("|" + "|".join(["---"] * len(headers)) + "|")

for model_name, m in rows:
    vals = [
        model_name,
        f"{m['Val Binary F1']:.4f}" if m["Val Binary F1"] is not None else "-",
        f"{m['Val Modality F1']:.4f}" if m["Val Modality F1"] is not None else "-",
        f"{m['Val Tech F1']:.4f}" if m["Val Tech F1"] is not None else "-",
        f"{m['Test Binary F1']:.4f}" if m["Test Binary F1"] is not None else "-",
        f"{m['Test Modality F1']:.4f}" if m["Test Modality F1"] is not None else "-",
        f"{m['Test Tech F1']:.4f}" if m["Test Tech F1"] is not None else "-",
    ]
    print("| " + " | ".join(vals) + " |")

print()
print("Baseline 참고:")
print("- Score-only MLP Test Tech F1: 0.5683")
print("- Embedding MLP Test Tech F1: 0.7849")
print("- Current AAT Test Tech F1: 0.7581")
print()