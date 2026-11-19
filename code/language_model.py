"""Model-agnostic n-gram language model (KenLM) for N-best ASR rescoring across any encoder backend."""
from typing import List, Optional, Sequence
import os
import re
import subprocess
import kenlm


class NgramLanguageModel:
    """Standalone n-gram LM wrapper for N-best hypothesis rescoring."""

    def __init__(
        self,
        klm_path: str,
        alpha: float = 0.15,
        beta: float = 0.0,
        char_level: bool = False,
    ) -> None:
        """
        Args:
            klm_path: Path to compiled KenLM binary (.klm) or ARPA file.
            alpha: Weight applied to the LM log10 score during rescoring.
            beta: Word insertion bonus (positive favors longer hypotheses, negative favors shorter).
            char_level: Score character sequences instead of word tokens.
        """
        if not klm_path:
            raise ValueError("klm_path must be a non-empty path to a KenLM .klm or .arpa file.")
        self.klm_path = klm_path
        self.alpha = alpha
        self.beta = beta
        self.char_level = char_level
        self.lm = kenlm.LanguageModel(klm_path)

    @staticmethod
    def normalize_text(text: str, char_level: bool = False) -> str:
        """Consistent Esperanto text normalization matching evaluation rules."""
        s = str(text).lower()
        s = re.sub(r"[^\w\sĉĝĥĵŝŭ]", " ", s, flags=re.UNICODE)
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            return ""
        if char_level:
            tokens: List[str] = []
            for word in s.split(" "):
                tokens.extend(list(word))
                tokens.append("<w>")
            return " ".join(tokens).strip()
        return s

    @classmethod
    def train_from_transcripts(
        cls,
        transcripts: Sequence[str],
        output_dir: str,
        model_name: str = "eo_ngram",
        order: int = 3,
        char_level: bool = False,
        alpha: float = 0.15,
        beta: float = 0.0,
    ) -> "NgramLanguageModel":
        """Build ARPA and binary KenLM files from any iterable of training transcripts."""
        os.makedirs(output_dir, exist_ok=True)
        suffix = "char" if char_level else "word"
        corpus_path = os.path.join(output_dir, f"{model_name}_{suffix}_corpus.txt")
        arpa_path = os.path.join(output_dir, f"{model_name}_{suffix}_{order}g.arpa")
        klm_path = os.path.join(output_dir, f"{model_name}_{suffix}_{order}g.klm")

        lines = [cls.normalize_text(t, char_level=char_level) for t in transcripts]
        lines = [line for line in lines if line]

        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        with open(corpus_path, "r", encoding="utf-8") as fin, open(arpa_path, "w", encoding="utf-8") as fout:
            subprocess.run(
                ["lmplz", "-o", str(order), "--discount_fallback"],
                stdin=fin,
                stdout=fout,
                check=True,
            )

        subprocess.run(["build_binary", arpa_path, klm_path], check=True)
        return cls(klm_path=klm_path, alpha=alpha, beta=beta, char_level=char_level)

    def score(self, text: str) -> float:
        """Return KenLM log10 probability for a hypothesis string."""
        norm = self.normalize_text(text, char_level=self.char_level)
        return float(self.lm.score(norm, bos=True, eos=True)) if norm else -100.0

    def rescore_candidates(
        self,
        candidates: Sequence[str],
        acoustic_scores: Optional[Sequence[float]] = None,
    ) -> str:
        """Select the highest-scoring candidate given acoustic scores and LM weight alpha."""
        if not candidates:
            return ""
        if len(candidates) == 1:
            return str(candidates[0]).strip()

        if acoustic_scores is None:
            acoustic_scores = [-0.5 * i for i in range(len(candidates))]

        best_cand = max(
            zip(candidates, acoustic_scores),
            key=lambda pair: (
                float(pair[1])
                + self.alpha * self.score(str(pair[0]))
                + self.beta * len(str(pair[0]).split())
            ),
        )[0]
        return str(best_cand).strip()

    def rescore_batch(
        self,
        batch_candidates: Sequence[Sequence[str]],
        batch_acoustic_scores: Optional[Sequence[Sequence[float]]] = None,
    ) -> List[str]:
        """Batch wrapper around rescore_candidates for evaluation loops."""
        results: List[str] = []
        for idx, cands in enumerate(batch_candidates):
            scores = batch_acoustic_scores[idx] if batch_acoustic_scores is not None else None
            results.append(self.rescore_candidates(cands, scores))
        return results