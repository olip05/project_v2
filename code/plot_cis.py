import argparse as argp
import json
import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import base_info as info


def compute_ci_95(values: List[float]) -> Tuple[float, float, float]:
    """Return (mean, lower_95_ci, upper_95_ci) in percentage units across runs."""
    arr = np.asarray(values, dtype=float) * 100.0
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    if se == 0.0:
        return mean, mean, mean
    lo, hi = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
    return mean, float(lo), float(hi)


def _format_model_label(model_name: str, custom_labels: Optional[Dict[str, str]] = None) -> str:
    if custom_labels and model_name in custom_labels:
        return custom_labels[model_name]
    if model_name == "openai_whisper-small":
        return "Whisper base"
    return (
        model_name.split("/")[-1]
        .replace("._models_openai_whisper-small_whisper-eo-", "Whisper ")
        .replace("vosk-model-small-eo-0.42", "Vosk small")
    )


def load_all_models_from_combined_json(
    json_path: str,
    strict_same_indices: bool = True,
) -> Tuple[List[str], Dict[str, List[float]], Dict[str, List[float]]]:
    """Extract per-run WER and CER series for every model in a combined AVG JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n_runs = int(data.get("runs", 0))
    run_keys = [str(i) for i in range(n_runs) if str(i) in data]
    if not run_keys:
        run_keys = sorted([k for k in data.keys() if k.isdigit()], key=int)
    if not run_keys:
        raise ValueError(f"No per-run entries found in {json_path}")

    if strict_same_indices:
        first_indices = list(map(int, data[run_keys[0]].get("indices", [])))
        for k in run_keys[1:]:
            cur_indices = list(map(int, data[k].get("indices", [])))
            if cur_indices != first_indices:
                raise ValueError(f"Indices mismatch between run '{run_keys[0]}' and run '{k}'.")

    candidate_models = data.get("models") or data[run_keys[0]].get("models") or []
    models: List[str] = []
    for m in candidate_models:
        if m not in ("models", "indices", "time") and all(m in data[k] for k in run_keys):
            if m not in models:
                models.append(m)

    if not models:
        for k in data[run_keys[0]]:
            if k not in ("models", "indices", "time") and isinstance(data[k], dict):
                if "wer" in data[k] and "cer" in data[k] and k not in models:
                    models.append(k)

    if not models:
        raise ValueError(f"No valid model metrics found in {json_path}")

    wers_by_model: Dict[str, List[float]] = {
        m: [float(data[k][m]["wer"]) for k in run_keys] for m in models
    }
    cers_by_model: Dict[str, List[float]] = {
        m: [float(data[k][m]["cer"]) for k in run_keys] for m in models
    }
    return models, wers_by_model, cers_by_model


def _plot_single_metric(
    labels: List[str],
    means: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    ylabel: str,
    title: str,
    color: str,
    output_path: str,
    mean_width: float = 0.36,
    cap_width: float = 0.16,
) -> None:
    """Plot a vertical whisker with shorter end caps and a long bar for the mean, starting at y=0."""
    x_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for x, m, lo, hi in zip(x_pos, means, lows, highs):
        lo_c = max(0.0, float(lo))
        hi_c = max(0.0, float(hi))
        m_c = max(0.0, float(m))

        if hi_c > lo_c:
            ax.vlines(x, lo_c, hi_c, colors=color, linewidth=0.5, zorder=2)
            ax.hlines(
                [lo_c, hi_c],
                x - cap_width / 2.0,
                x + cap_width / 2.0,
                colors=color,
                linewidth=0.7,
                zorder=3,
            )

        ax.hlines(
            m_c,
            x - mean_width / 2.0,
            x + mean_width / 2.0,
            colors=color,
            linewidth=1,
            zorder=4,
        )

    ax.set_ylim(bottom=0, top=max(100, max(highs)))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_separate_confidence_intervals(
    eval_json_path: str,
    wer_output_path: str = "wer_ci_comparison.png",
    cer_output_path: str = "cer_ci_comparison.png",
    strict_same_indices: bool = True,
    custom_labels: Optional[Dict[str, str]] = None,
) -> None:
    """Generate two separate CI plots (WER and CER) for all models in a combined JSON."""
    models, wers_by_model, cers_by_model = load_all_models_from_combined_json(
        eval_json_path,
        strict_same_indices=strict_same_indices,
    )

    records: List[Dict[str, object]] = []
    for m in models:
        m_wer, lo_wer, hi_wer = compute_ci_95(wers_by_model[m])
        m_cer, lo_cer, hi_cer = compute_ci_95(cers_by_model[m])
        records.append({
            "label": _format_model_label(m, custom_labels=custom_labels),
            "mean_wer": m_wer,
            "lo_wer": lo_wer,
            "hi_wer": hi_wer,
            "mean_cer": m_cer,
            "lo_cer": lo_cer,
            "hi_cer": hi_cer,
        })

    labels = [str(r["label"]) for r in records]

    _plot_single_metric(
        labels=labels,
        means=np.array([float(r["mean_wer"]) for r in records]),
        lows=np.array([float(r["lo_wer"]) for r in records]),
        highs=np.array([float(r["hi_wer"]) for r in records]),
        ylabel="WER [%]",
        title="Word Error Rate across models",
        color="#1f77b4",
        output_path=wer_output_path,
    )

    _plot_single_metric(
        labels=labels,
        means=np.array([float(r["mean_cer"]) for r in records]),
        lows=np.array([float(r["lo_cer"]) for r in records]),
        highs=np.array([float(r["hi_cer"]) for r in records]),
        ylabel="CER [%]",
        title="Character Error Rate across models",
        color="#d62728",
        output_path=cer_output_path,
    )


if __name__ == "__main__":
    parser = argp.ArgumentParser(description="Plot combined results")
    parser.add_argument("--file", default="", type=str, help="the file name to analyse")
    args = parser.parse_args()

    plot_separate_confidence_intervals(
        eval_json_path=os.path.join(info.EVALUATION_PATH, args.file),
        wer_output_path=os.path.join(info.EVALUATION_PATH, "wer_ci_comparison.png"),
        cer_output_path=os.path.join(info.EVALUATION_PATH, "cer_ci_comparison.png"),
        strict_same_indices=True,
    )