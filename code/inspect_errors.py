from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import os
import jiwer
import argparse as argp
import pandas as pd
import base_info as info


def load_ground_truth_by_index(
    dataset_path: str,
    transcript_col: str = "transcript",
) -> Dict[int, str]:
    """Load dataset rows keyed by DataFrame index so `indices` map directly to references."""
    df = pd.read_csv(dataset_path)
    return {int(idx): str(row[transcript_col]) for idx, row in df.iterrows()}


def _normalize_text(text: str) -> str:
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
    ])
    return transform(text)


def analyze_run(
    references: Sequence[str],
    predictions: Sequence[str],
    indices: Optional[Sequence[int]] = None,
    top_k: int = 15,
) -> Dict[str, Any]:
    """Analyze word/character error types, confusions, and worst samples for one run."""
    assert len(references) == len(predictions), (
        f"Length mismatch: {len(references)} references vs {len(predictions)} predictions"
    )

    norm_refs = [_normalize_text(r) for r in references]
    norm_preds = [_normalize_text(p) for p in predictions]

    words_preds = len({
        word
        for text in norm_preds
        for word in text.split()
        if word
    })
    chars_preds = len({
        ch
        for text in norm_preds
        for ch in text
    })

    words_refs = len({
        word
        for text in norm_refs
        for word in text.split()
        if word
    })
    chars_refs = len({
        ch
        for text in norm_refs
        for ch in text
    })

    word_out = jiwer.process_words(norm_refs, norm_preds)
    char_out = jiwer.process_characters(norm_refs, norm_preds)

    # Word-level error counts and confusions
    sub_pairs: Counter[Tuple[str, str]] = Counter()
    deleted_words: Counter[str] = Counter()
    inserted_words: Counter[str] = Counter()

    for ref_words, pred_words, chunks in zip(
        word_out.references, word_out.hypotheses, word_out.alignments
    ):
        for chunk in chunks:
            if chunk.type == "substitute":
                for r_w, h_w in zip(
                    ref_words[chunk.ref_start_idx : chunk.ref_end_idx],
                    pred_words[chunk.hyp_start_idx : chunk.hyp_end_idx],
                ):
                    sub_pairs[(r_w, h_w)] += 1
            elif chunk.type == "delete":
                for r_w in ref_words[chunk.ref_start_idx : chunk.ref_end_idx]:
                    deleted_words[r_w] += 1
            elif chunk.type == "insert":
                for h_w in pred_words[chunk.hyp_start_idx : chunk.hyp_end_idx]:
                    inserted_words[h_w] += 1

    # Standard character-level substitutions, deletions, and insertions
    char_sub_pairs: Counter[Tuple[str, str]] = Counter()
    deleted_chars: Counter[str] = Counter()
    inserted_chars: Counter[str] = Counter()

    for ref_chars, pred_chars, chunks in zip(
        char_out.references, char_out.hypotheses, char_out.alignments
    ):
        for chunk in chunks:
            if chunk.type == "substitute":
                for r_c, h_c in zip(
                    ref_chars[chunk.ref_start_idx : chunk.ref_end_idx],
                    pred_chars[chunk.hyp_start_idx : chunk.hyp_end_idx],
                ):
                    char_sub_pairs[(r_c, h_c)] += 1
            elif chunk.type == "delete":
                for r_c in ref_chars[chunk.ref_start_idx : chunk.ref_end_idx]:
                    deleted_chars[r_c] += 1
            elif chunk.type == "insert":
                for h_c in pred_chars[chunk.hyp_start_idx : chunk.hyp_end_idx]:
                    inserted_chars[h_c] += 1

    # Per-sample breakdown to spot hardest dataset indices
    sample_rows = []
    for pos, (r_norm, h_norm) in enumerate(zip(norm_refs, norm_preds)):
        ds_idx = int(indices[pos]) if indices is not None and pos < len(indices) else pos
        w_res = jiwer.process_words(r_norm, h_norm)
        c_res = jiwer.process_characters(r_norm, h_norm)
        sample_rows.append({
            "dataset_idx": ds_idx,
            "wer": round(float(w_res.wer), 4),
            "cer": round(float(c_res.cer), 4),
            "subs": w_res.substitutions,
            "dels": w_res.deletions,
            "ins": w_res.insertions,
            "ref": r_norm,
            "hyp": h_norm,
        })

    worst_df = pd.DataFrame(sample_rows).sort_values(
        by=["wer", "cer", "subs", "dels", "ins"], ascending=False
    )

    return {
        "word_counts": {
            "total_refs": words_refs,
            "total_preds": words_preds,
            "wer": round(float(word_out.wer), 4),
            "substitutions": word_out.substitutions,
            "deletions": word_out.deletions,
            "insertions": word_out.insertions,
            "hits": word_out.hits,
            "total": word_out.hits + word_out.substitutions + word_out.deletions,
        },
        "char_counts": {
            "total_refs": chars_refs,
            "total_preds": chars_preds,
            "cer": round(float(char_out.cer), 4),
            "substitutions": char_out.substitutions,
            "deletions": char_out.deletions,
            "insertions": char_out.insertions,
            "hits": char_out.hits,
            "total": char_out.hits + char_out.substitutions + char_out.deletions,
        },
        "top_word_substitutions": sub_pairs.most_common(top_k),
        "top_deleted_words": deleted_words.most_common(top_k),
        "top_inserted_words": inserted_words.most_common(top_k),
        "top_char_substitutions": char_sub_pairs.most_common(top_k),
        "top_deleted_chars": deleted_chars.most_common(top_k),
        "top_inserted_chars": inserted_chars.most_common(top_k),
        "worst_samples": worst_df.head(top_k),
    }


def analyze_eval_json(
    eval_json_path: str,
    dataset_csv_path: str,
    run_index: int = -1,
    aggregate_all_runs: bool = True,
    top_k: int = 15,
) -> Dict[str, Any]:
    """Load an evaluation JSON file and analyze a single run (default: latest) or all runs combined."""
    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ref_lookup = load_ground_truth_by_index(dataset_csv_path)
    runs: List[Dict[str, Any]] = [data.get(f"{r}") for r in range(data.get("runs", 0)) if data.get(f"{r}") is not None]
    assert len(runs) > 0, "No runs found in evaluation JSON"

    model_key = "." + eval_json_path.split("-.")[1].replace("-AVG.json", "")

    if aggregate_all_runs:
        all_refs: List[str] = []
        all_preds: List[str] = []
        all_indices: List[int] = []
        for r in runs:
            idxs = r["indices"]
            preds = r[model_key]["predictions"]
            all_refs.extend([ref_lookup[int(i)] for i in idxs])
            all_preds.extend(preds)
            all_indices.extend([int(i) for i in idxs])
        return analyze_run(all_refs, all_preds, indices=all_indices, top_k=top_k)

    selected = runs[run_index]
    idxs = selected["indices"]
    refs = [ref_lookup[int(i)] for i in idxs]
    preds = selected[model_key]["predictions"]
    return analyze_run(refs, preds, indices=idxs, top_k=top_k)


def print_report(report: Dict[str, Any]) -> None:
    wc = report["word_counts"]
    cc = report["char_counts"]

    print(
        f"WER: {wc['wer']*100:.2f}% | Subs: {wc['substitutions']} | "
        f"Dels: {wc['deletions']} | Ins: {wc['insertions']} | Total words: {wc['total']}"
    )
    print(f"Word count: Refs: {wc["total_refs"]} | Predictions: {wc["total_preds"]}")
    print(
        f"CER: {cc['cer']*100:.2f}% | Subs: {cc['substitutions']} | "
        f"Dels: {cc['deletions']} | Ins: {cc['insertions']} | Total chars: {cc['total']}"
    )
    print(f"Char count: Refs: {cc["total_refs"]} | Predictions: {cc["total_preds"]}")

    print("\nTop Word Substitutions (ref -> hyp):")
    for (r_w, h_w), cnt in report["top_word_substitutions"]:
        print(f"  {cnt:4d}x  '{r_w}' -> '{h_w}'")

    print("\nTop Deleted Words:")
    for w, cnt in report["top_deleted_words"]:
        print(f"  {cnt:4d}x  '{w}'")

    print("\nTop Inserted Words:")
    for w, cnt in report["top_inserted_words"]:
        print(f"  {cnt:4d}x  '{w}'")

    print("\nTop Character Substitutions (ref -> hyp):")
    for (r_c, h_c), cnt in report["top_char_substitutions"]:
        print(f"  {cnt:4d}x  '{r_c}' -> '{h_c}'")

    print("\nTop Deleted Characters:")
    for c, cnt in report["top_deleted_chars"]:
        print(f"  {cnt:4d}x  '{c}'")

    print("\nTop Inserted Characters:")
    for c, cnt in report["top_inserted_chars"]:
        print(f"  {cnt:4d}x  '{c}'")

    print("\nWorst Samples:")
    print(report["worst_samples"].to_string(index=False))


if __name__ == "__main__":
    parser = argp.ArgumentParser(description="Analyze ASR evaluation errors at word and character levels.")
    parser.add_argument("--eval", default="", type=str, help="the file name to analyse")
    args = parser.parse_args()

    dataset_path = os.path.join(info.DATASET_PATH, "geo", "dev.csv")

    report = analyze_eval_json(
        eval_json_path=os.path.join(info.EVALUATION_PATH, args.eval),
        dataset_csv_path=dataset_path,
        aggregate_all_runs=True,
        top_k=15,
    )
    print_report(report)