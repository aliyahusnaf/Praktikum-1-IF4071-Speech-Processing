import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import VOWEL_ORDER
from .preprocessing import DEFAULT_DATA_DIR

SYSTEMS = {"DTW": "dtw_results.json", "Gaussian": "gaussian_results.json"}
SCENARIOS = {"closed_recordings": "Closed Recordings", "closed_speakers": "Closed Speakers",
             "open_speakers": "Open Speakers"}


def load_results(output_dir):
    return {system: json.loads((Path(output_dir) / name).read_text(encoding="utf-8"))["results"]
            for system, name in SYSTEMS.items()}


def fmt(value):
    return f"{value:.2f}%"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def with_average(label, values):
    return [label] + [fmt(v) for v in values] + [fmt(np.mean(values))]


def summary_table(results):
    rows = [with_average(system, [res[s]["accuracy"] for s in SCENARIOS])
            for system, res in results.items()]
    return table(["Sistem"] + list(SCENARIOS.values()) + ["Rata-rata"], rows)


def vowel_table(results):
    columns = [(system, s) for system in results for s in SCENARIOS]
    header = ["Vowel"] + [f"{system} {SCENARIOS[s]}" for system, s in columns]
    rows = [[v] + [fmt(results[system][s]["per_vowel"][v]["accuracy"]) for system, s in columns]
            for v in VOWEL_ORDER]
    rows.append(["Average Accuracy"] + [fmt(results[system][s]["accuracy"]) for system, s in columns])
    return table(header, rows)


def speaker_table(results, scenarios):
    columns = [(system, s) for system in results for s in scenarios]
    speakers = list(results["DTW"][scenarios[0]]["per_speaker"])
    rows = [with_average(speaker, [results[system][s]["per_speaker"][speaker]["accuracy"]
                                   for system, s in columns])
            for speaker in speakers]
    rows.append(with_average("Average", [results[system][s]["accuracy"] for system, s in columns]))
    header = ["Speakers"] + [f"{system} {SCENARIOS[s]}" for system, s in columns] + ["Average"]
    return table(header, rows)


def confusion_table(result):
    labels = result["labels"]
    rows = [[t] + [str(result["confusion"][t][p]) for p in labels] for t in labels]
    return table(["asli \\ prediksi"] + labels, rows)


def build_report(results):
    parts = ["# Perbandingan DTW dan Model Gaussian", "## Ringkasan akurasi", summary_table(results),
             "## Tabel 1. Accuracy untuk setiap vowel", vowel_table(results),
             "## Tabel 2. Closed experiments (per speaker)",
             speaker_table(results, ["closed_recordings", "closed_speakers"]),
             "## Tabel 3. Open experiments (per speaker)",
             speaker_table(results, ["open_speakers"]),
             "## Confusion matrix"]
    for system, res in results.items():
        for s, title in SCENARIOS.items():
            parts += [f"### {system}: {title}", confusion_table(res[s])]
    return "\n\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Tabel perbandingan DTW dan Gaussian")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA_DIR.parents[1] / "outputs")
    args = parser.parse_args()
    report = build_report(load_results(args.output_dir))
    path = args.output_dir / "comparison.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Disimpan ke {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
