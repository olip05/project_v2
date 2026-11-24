import argparse as argp
import json
import os
from typing import Dict, List, Optional, Tuple
from matplotlib.lines import Line2D
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


def _draw_metric_intervals(
    ax: plt.Axes,
    x_positions: np.ndarray,
    means: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    color: str,
    mean_width: float = 0.36,
    cap_width: float = 0.16,
    whisker_lw: float = 0.5,
    cap_lw: float = 0.7,
    mean_lw: float = 1.0,
) -> None:
    """Draw a vertical whisker with shorter end caps and a long bar for the mean."""
    for x, m, lo, hi in zip(x_positions, means, lows, highs):
        lo_c = max(0.0, float(lo))
        hi_c = max(0.0, float(hi))
        m_c = max(0.0, float(m))

        if hi_c > lo_c:
            ax.vlines(x, lo_c, hi_c, colors=color, linewidth=whisker_lw, zorder=2)
            ax.hlines(
                [lo_c, hi_c],
                x - cap_width / 2.0,
                x + cap_width / 2.0,
                colors=color,
                linewidth=cap_lw,
                zorder=3,
            )

        ax.hlines(
            m_c,
            x - mean_width / 2.0,
            x + mean_width / 2.0,
            colors=color,
            linewidth=mean_lw,
            zorder=4,
        )


def _plot_single_metric(
    labels: List[str],
    means: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    ylabel: str,
    title: str,
    color: str,
    output_path: str,
) -> None:
    """Create and save a single-metric plot starting at y=0."""
    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 5))

    _draw_metric_intervals(
        ax=ax,
        x_positions=x_pos,
        means=means,
        lows=lows,
        highs=highs,
        color=color,
    )

    ax.set_ylim(bottom=0, top=max(100.0, float(np.max(highs)) + 2.0))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close(fig)
    print(f"Saved {output_path}")


def _plot_combined_metric(
    labels: List[str],
    means_wer: np.ndarray,
    lows_wer: np.ndarray,
    highs_wer: np.ndarray,
    means_cer: np.ndarray,
    lows_cer: np.ndarray,
    highs_cer: np.ndarray,
    title: str,
    output_path: str,
    offset: float = 0.14,
) -> None:
    """Create and save a combined plot showing WER and CER side by side per model."""
    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    _draw_metric_intervals(
        ax=ax,
        x_positions=x_pos,
        means=means_wer,
        lows=lows_wer,
        highs=highs_wer,
        color="#1f77b4",
        mean_width=0.36,
        cap_width=0.16,
        whisker_lw=0.5,
        cap_lw=0.7,
        mean_lw=1,
    )
    _draw_metric_intervals(
        ax=ax,
        x_positions=x_pos,
        means=means_cer,
        lows=lows_cer,
        highs=highs_cer,
        color="#d62728",
        mean_width=0.36,
        cap_width=0.16,
        whisker_lw=0.5,
        cap_lw=0.7,
        mean_lw=1,
    )

    max_top = max(100.0, float(np.max(highs_wer)) + 2.0, float(np.max(highs_cer)) + 2.0)
    ax.set_ylim(bottom=0, top=max_top)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10.5)
    ax.set_ylabel("Error Rate [%]", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", lw=2.2, label="WER"),
        Line2D([0], [0], color="#d62728", lw=2.2, label="CER"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_confidence_intervals(
    eval_json_path: str,
    wer_output_path: str = "wer_ci_comparison.png",
    cer_output_path: str = "cer_ci_comparison.png",
    combined_output_path: str = "wer_cer_ci_comparison.png",
    strict_same_indices: bool = False,
    custom_labels: Optional[Dict[str, str]] = None,
) -> None:
    """Generate separate WER and CER plots as well as a combined WER+CER plot."""
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
    means_w = np.array([float(r["mean_wer"]) for r in records])
    lows_w = np.array([float(r["lo_wer"]) for r in records])
    highs_w = np.array([float(r["hi_wer"]) for r in records])

    means_c = np.array([float(r["mean_cer"]) for r in records])
    lows_c = np.array([float(r["lo_cer"]) for r in records])
    highs_c = np.array([float(r["hi_cer"]) for r in records])

    _plot_single_metric(
        labels=labels,
        means=means_w,
        lows=lows_w,
        highs=highs_w,
        ylabel="WER [%]",
        title="Word Error Rate across models",
        color="#1f77b4",
        output_path=wer_output_path,
    )

    _plot_single_metric(
        labels=labels,
        means=means_c,
        lows=lows_c,
        highs=highs_c,
        ylabel="CER [%]",
        title="Character Error Rate across models",
        color="#d62728",
        output_path=cer_output_path,
    )

    _plot_combined_metric(
        labels=labels,
        means_wer=means_w,
        lows_wer=lows_w,
        highs_wer=highs_w,
        means_cer=means_c,
        lows_cer=lows_c,
        highs_cer=highs_c,
        title="Word and Character error rate across models",
        output_path=combined_output_path,
    )


if __name__ == "__main__":
    parser = argp.ArgumentParser(description="Plot combined results")
    parser.add_argument("--file", default="", type=str, help="the file name to analyse")
    args = parser.parse_args()

    plot_confidence_intervals(
        eval_json_path=os.path.join(info.EVALUATION_PATH, args.file),
        wer_output_path=os.path.join(info.EVALUATION_PATH, "wer_ci_comparison.png"),
        cer_output_path=os.path.join(info.EVALUATION_PATH, "cer_ci_comparison.png"),
        combined_output_path=os.path.join(info.EVALUATION_PATH, "wer_cer_ci_comparison.png"),
        strict_same_indices=False,
    )
