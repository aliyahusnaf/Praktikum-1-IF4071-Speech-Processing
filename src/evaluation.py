# Perhitungan akurasi: total, per vowel, per speaker, dan confusion matrix.
# Dipakai oleh sistem Gaussian, dan bisa dipakai juga oleh sistem DTW.

VOWEL_ORDER = ["a", "i", "u", "e", "o"]


def accuracy(correct, total):
    # Accuracy = jumlah prediksi benar / jumlah data uji x 100%
    return 100.0 * correct / total if total else float("nan")


def evaluate(records, predictions):
    # records: list dict berisi "label", "speaker", "filename"
    # predictions: list label hasil prediksi, urutannya sama dengan records
    if len(records) != len(predictions):
        raise ValueError("Jumlah prediksi harus sama dengan jumlah data uji")
    rows = [{"filename": r["filename"], "speaker": r["speaker"], "label": r["label"],
             "prediction": p, "correct": p == r["label"]}
            for r, p in zip(records, predictions)]
    labels = VOWEL_ORDER + sorted({r["label"] for r in rows} - set(VOWEL_ORDER))
    confusion = {t: {p: 0 for p in labels} for t in labels}
    per_vowel, per_speaker = {}, {}
    for row in rows:
        confusion[row["label"]][row["prediction"]] += 1
        for table, key in [(per_vowel, row["label"]), (per_speaker, row["speaker"])]:
            stat = table.setdefault(key, {"correct": 0, "total": 0})
            stat["correct"] += row["correct"]
            stat["total"] += 1
    for table in (per_vowel, per_speaker):
        for stat in table.values():
            stat["accuracy"] = accuracy(stat["correct"], stat["total"])
    correct = sum(r["correct"] for r in rows)
    return {"accuracy": accuracy(correct, len(rows)), "correct": correct, "total": len(rows),
            "per_vowel": {v: per_vowel[v] for v in labels if v in per_vowel},
            "per_speaker": dict(sorted(per_speaker.items())),
            "confusion": confusion, "labels": labels, "rows": rows}


def format_confusion(result):
    # Teks confusion matrix: baris = label asli, kolom = prediksi.
    labels = result["labels"]
    lines = ["asli\\pred " + " ".join(f"{p:>3}" for p in labels)]
    for t in labels:
        lines.append(f"{t:>9} " + " ".join(f"{result['confusion'][t][p]:>3}" for p in labels))
    return "\n".join(lines)
