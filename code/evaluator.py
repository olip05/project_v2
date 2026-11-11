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
        cer = self.compute_cer(predictions)

        return (model.name(), {'predictions': predictions, 'wer': wer, 'cer': cer})

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

        model_names = [m.name() for m in models]
        averages: Dict[str, Dict[str, float]] = {}

        for m_name in model_names:
            # FIXME oplew add std_dev to calculations
            wers = [res[str(i)][m_name]["wer"] for i in range(runs)]
            cers = [res[str(i)][m_name]["cer"] for i in range(runs)]
            averages[m_name] = {
                "avg_wer": float(np.mean(wers)),
                "avg_cer": float(np.mean(cers))
            }

        res["models"] = [x.split("/")[-1] for x in list(res["0"].keys())]
        res["averages"] = averages
        res["time"] = str(datetime.datetime.now())

        self.print_avg_results(res)

        self.results_to_file(res, True)
        return res

    def print_single_results(self, res: Dict[str, Any]):
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
            print(f"|   CER: {res[model_key]['cer'] * 100:>8.2f} %"
                f"{'':<{width - 20}}|")
            print("|" + " " * (width - 2) + "|")

        print("+" + "-" * (width - 2) + "+")

    def print_avg_results(self, res: Dict[str, Any]):
        width = 124

        print("+" + "-" * (width - 2) + "+")
        print(f"| Time: {res['time']:<{width - 9}}|")
        print(f"| Test size: {len(self.indices):<{width - 14}}|")
        print("+" + "-" * (width - 2) + "+")

        for m in res["averages"].keys():

            print(f"| Model: {m:<{width - 10}}|")
            print(f"|   WER: {res["averages"][m]['avg_wer'] * 100:>8.2f} %"
                f"{'':<{width - 20}}|")
            print(f"|   CER: {res["averages"][m]['avg_cer'] * 100:>8.2f} %"
                f"{'':<{width - 20}}|")
            print("|" + " " * (width - 2) + "|")

        print("+" + "-" * (width - 2) + "+")
        
