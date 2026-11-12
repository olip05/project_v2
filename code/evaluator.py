"""evaluation"""
from typing import List, Dict, Tuple, Any
import json
import os
import datetime
import jiwer
import pandas as pd
import numpy as np
import librosa

from model import Model
import base_info as info

class Evaluator:
    """"run evaluation with WER and CER"""
    def __init__(self, test_data: pd.DataFrame) -> None:
        self.test_data = test_data
        self.sample_test_data()

    def sample_test_data(self) -> None:
        """update random inputs"""
        self.indices = np.random.choice(len(self.test_data), size=min(len(self.test_data), info.EVALUATION_SAMPLES), replace=False)
        self.reference = [str(self.test_data["transcript"].iloc[i]) for i in self.indices]

    def get_predictions(self, m: Model) -> List[str]:
        """run prediction on given test samples"""
        inputs: List[Any] = []
        for i in self.indices:
            file_path = self.test_data["file"].iloc[i]
            wav, _ = librosa.load(os.path.join(info.DATASET_PATH, file_path), sr=info.SR)
            inputs.append(wav)

        return m.predict(m.format_inputs(inputs))

    def compute_wer(self, predictions: List[str]) -> float:
        """compute WER"""

        transformation = jiwer.Compose([
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords()
        ])

        return float(jiwer.wer(
            self.reference,
            predictions,
            reference_transform=transformation,
            hypothesis_transform=transformation
        ))

    def compute_cer(self, predictions: List[str]) -> float:
        """compute CER"""

        transformation = jiwer.Compose([
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfChars()
        ])

        return float(jiwer.cer(
            self.reference,
            predictions,
            reference_transform=transformation,
            hypothesis_transform=transformation
        ))

    def compute_metrics(self, model: Model) -> Tuple[str, Any]:
        """compute all defined metrices"""
        predictions: List[str] = self.get_predictions(model)

        wer = self.compute_wer(predictions)
        error_words = self.compute_detailed_errors(predictions, True)
        cer = self.compute_cer(predictions)
        error_chars = self.compute_detailed_errors(predictions, False)

        return (model.name(), {'predictions': predictions, 'wer': wer, 'word_errors': error_words, 'cer': cer, 'char_errors': error_chars})

    def compute_detailed_errors(self, predictions: List[str], words: bool) -> Dict[str, Any]:
        """Compute detailed word error breakdown (Substitutions, Deletions, Insertions, Hits)."""
        transformation = jiwer.Compose([
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            (jiwer.ReduceToListOfListOfWords() if words else jiwer.ReduceToListOfListOfChars())
        ])

        if words:
            out = jiwer.process_words(
                self.reference,
                predictions,
                reference_transform=transformation,
                hypothesis_transform=transformation
            )
        else:
            out = jiwer.process_characters(
                self.reference,
                predictions,
                reference_transform=transformation,
                hypothesis_transform=transformation
            )

        return {
            "substitutions": int(out.substitutions),
            "deletions": int(out.deletions),
            "insertions": int(out.insertions),
            "hits": int(out.hits),
            "total": int(out.substitutions + out.deletions + out.hits)
        }

    def results_to_file(self, content: Dict[str, Any], avg: bool) -> None:
        """save results fo json"""
        with open(os.path.join(info.EVALUATION_PATH, f"{str(content['time']).replace(' ', '_')}-{'+'.join(content["models"])}{'-AVG' if avg else ''}.json"), "w", encoding="utf8") as f:
            json.dump(content, f, indent=4)

    def compare_models(self, models: List[Model], write_to_file: bool = False, part_of_avg: bool = False) -> Dict[str, Any]:
        """run evaluation for all models"""
        res: Dict[str, Any] = {}
        for m in models:
            name, metrics = self.compute_metrics(m)
            res[name] = metrics

        res["models"] = [x.split("/")[-1] for x in list(res.keys())]
        res["indices"] = list(self.indices.tolist())
        res["time"] = str(datetime.datetime.now())

        if write_to_file:
            self.results_to_file(res, False)

        if not part_of_avg:
            self.print_single_results(res)

        return res

    def multiple_runs(self, models: List[Model], runs: int = 30) -> Dict[str, Dict[str, Any]]:
        """get statistical results"""
        res: Dict[str, Any] = {}
        for i in range(runs):
            res[str(i)] = self.compare_models(models, part_of_avg=True)
            self.sample_test_data()
            print(f"{datetime.datetime.now().time()} Runs completed: {i}")

        model_names = [m.name() for m in models]
        averages: Dict[str, Any] = {}

        for m_name in model_names:
            wers = [res[str(i)][m_name]["wer"] for i in range(runs)]
            cers = [res[str(i)][m_name]["cer"] for i in range(runs)]

            total_word_subs = sum(res[str(i)][m_name]["word_errors"]["substitutions"] for i in range(runs))
            total_word_dels = sum(res[str(i)][m_name]["word_errors"]["deletions"] for i in range(runs))
            total_word_inss = sum(res[str(i)][m_name]["word_errors"]["insertions"] for i in range(runs))
            total_word_hits = sum(res[str(i)][m_name]["word_errors"]["hits"] for i in range(runs))
            total_word_total = sum(res[str(i)][m_name]["word_errors"]["total"] for i in range(runs))

            total_char_subs = sum(res[str(i)][m_name]["char_errors"]["substitutions"] for i in range(runs))
            total_char_dels = sum(res[str(i)][m_name]["char_errors"]["deletions"] for i in range(runs))
            total_char_inss = sum(res[str(i)][m_name]["char_errors"]["insertions"] for i in range(runs))
            total_char_hits = sum(res[str(i)][m_name]["char_errors"]["hits"] for i in range(runs))
            total_char_total = sum(res[str(i)][m_name]["char_errors"]["total"] for i in range(runs))

            averages[m_name] = {
                "min_wer": float(np.min(wers)),
                "avg_wer": float(np.mean(wers)),
                "median_wer": float(np.median(wers)),
                "std_wer": float(np.std(wers)),
                "max_wer": float(np.max(wers)),
                "min_cer": float(np.min(cers)),
                "avg_cer": float(np.mean(cers)),
                "median_cer": float(np.median(cers)),
                "std_cer": float(np.std(cers)),
                "max_cer": float(np.max(cers)),
                "wer_errors": {
                    "substitutions": total_word_subs,
                    "deletions": total_word_dels,
                    "insertions": total_word_inss,
                    "hits": total_word_hits,
                    "total": total_word_total
                },
                "cer_errors": {
                    "substitutions": total_char_subs,
                    "deletions": total_char_dels,
                    "insertions": total_char_inss,
                    "hits": total_char_hits,
                    "total": total_char_total
                }
            }

        res["models"] = [x.split("/")[-1] for x in list(res["0"].keys()) if x not in ("models", "indices", "time")]
        res["averages"] = averages
        res["time"] = str(datetime.datetime.now())
        res["runs"] = runs

        self.print_avg_results(res)
        self.results_to_file(res, True)
        return res

    def print_single_results(self, res: Dict[str, Any]):
        """output results for one run"""
        width = 124

        print("+" + "-" * (width - 2) + "+")
        print(f"| Time: {res['time']:<{width - 9}}|")
        print(f"| Test size: {len(res['indices']):<{width - 14}}|")
        print("+" + "-" * (width - 2) + "+")

        for m in res["models"]:
            # Find the corresponding model result
            model_key = next(
                key for key in res
                if key not in ("models", "indices", "time")
                and key.split("/")[-1] == m
            )

            print(f"| Model: {m:<{width - 10}}|")
            print(f"|   WER: {res[model_key]['wer'] * 100:>8.2f} %"
                f"{'':<{width - 20}}|")
            print(f"|   Total: {res[model_key]["word_errors"]['total']}, Subs: {res[model_key]["word_errors"]['substitutions']}, Dels: {res[model_key]["word_errors"]['deletions']}, Ins: {res[model_key]["word_errors"]['insertions']}{'':<{width - 50}}|")
            print(f"|   CER: {res[model_key]['cer'] * 100:>8.2f} %"
                f"{'':<{width - 20}}|")
            print(f"|   Total: {res[model_key]["char_errors"]['total']}, Subs: {res[model_key]["char_errors"]['substitutions']}, Dels: {res[model_key]["char_errors"]['deletions']}, Ins: {res[model_key]["char_errors"]['insertions']}{'':<{width - 50}}|")
            print("|" + " " * (width - 2) + "|")

        print("+" + "-" * (width - 2) + "+")

    def print_avg_results(self, res: Dict[str, Any]):
        """output averaged results"""
        width = 124

        print("+" + "-" * (width - 2) + "+")
        print(f"| Time: {res['time']:<{width - 9}}|")
        print(f"| Test size: {len(self.indices):<{width - 14}}|")
        print(f"| Runs: {res["runs"]:<{width - 10}}|")
        print("+" + "-" * (width - 2) + "+")

        for m in res["averages"].keys():

            print(f"| Model: {m:<{width - 10}}|")
            print(f"|   WER: min: {res["averages"][m]['min_wer'] * 100:>4.2f} %, avg: {res["averages"][m]['avg_wer'] * 100:>4.2f} %, std: {res["averages"][m]['std_wer'] * 100:>4.2f} %, max: {res["averages"][m]['max_wer'] * 100:>4.2f} %" f"{'':<{width - 45}}|")
            print(f"|   Total: {res["averages"][m]["wer_errors"]['total']}, Subs: {res["averages"][m]["wer_errors"]['substitutions']}, Dels: {res["averages"][m]["wer_errors"]['deletions']}, Ins: {res["averages"][m]["wer_errors"]['insertions']}{'':<{width - 50}}|")
            print(f"|   CER: min: {res["averages"][m]['min_cer'] * 100:>4.2f} %, avg: {res["averages"][m]['avg_cer'] * 100:>4.2f} %, std: {res["averages"][m]['std_cer'] * 100:>4.2f}%, max: {res["averages"][m]['max_cer'] * 100:>4.2f} %" f"{'':<{width - 45}}|")
            print(f"|   Total: {res["averages"][m]["cer_errors"]['total']}, Subs: {res["averages"][m]["cer_errors"]['substitutions']}, Dels: {res["averages"][m]["cer_errors"]['deletions']}, Ins: {res["averages"][m]["cer_errors"]['insertions']}{'':<{width - 50}}|")

            print("|" + " " * (width - 2) + "|")

        print("+" + "-" * (width - 2) + "+")
