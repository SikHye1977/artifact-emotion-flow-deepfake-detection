import re
from pathlib import Path

LOGS = {
    "Score-only MLP": "logs_compare/zeroshot_avdf1m_score.log",
    "Embedding MLP": "logs_compare/zeroshot_avdf1m_embedding.log",
    "AAT old": "logs_compare/zeroshot_avdf1m_aat.log",
    "AAT small tuned": "logs_compare/zeroshot_avdf1m_aat_small.log",
}

def extract_summary(text, metric):
    pattern = rf"{re.escape(metric)}:\s+([0-9.]+)"
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

rows = []

for name, path in LOGS.items():
    p = Path(path)
    if not p.exists():
        print(f"[WARN] missing: {path}")
        continue

    text = p.read_text(errors="ignore")

    rows.append({
        "model": name,
        "binary_acc": extract_summary(text, "Binary Acc"),
        "binary_f1": extract_summary(text, "Binary Macro-F1"),
        "modality_acc": extract_summary(text, "Modality Acc"),
        "modality_f1": extract_summary(text, "Modality Macro-F1"),
    })

print()
print("| Model | Binary Acc | Binary Macro-F1 | Modality Acc | Modality Macro-F1 |")
print("|---|---:|---:|---:|---:|")

for r in rows:
    print(
        f"| {r['model']} | "
        f"{r['binary_acc']:.4f} | "
        f"{r['binary_f1']:.4f} | "
        f"{r['modality_acc']:.4f} | "
        f"{r['modality_f1']:.4f} |"
    )
