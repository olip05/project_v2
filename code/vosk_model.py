"""Vosk model"""
from typing import List, Any, Dict
import os
import json
import pandas as pd
import numpy as np
from vosk import Model as VoskNativeModel, KaldiRecognizer # type: ignore
from model import Model
import base_info as info

class VoskModel(Model):
    """Vosk model wrapper implementing the Model interface"""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name=model_name)
        self.is_trained: bool = False
        self.vosk_model = VoskNativeModel(os.path.join(info.MODEL_PATH, model_name))

    def format_inputs(self, inputs: List[Any]) -> List[bytes]:
        pcm_bytes_list: List[bytes] = []
        for audio in inputs:
            audio_np = np.asarray(audio, dtype=np.float32)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            pcm_bytes_list.append(audio_int16.tobytes())
        return pcm_bytes_list

    def load(self, weight_path: str) -> None:
        self.vosk_model = VoskNativeModel(os.path.join(info.MODEL_PATH, weight_path))
        self.is_trained = True

    def predict(self, inputs: Any) -> List[str]:
        transcriptions: List[str] = []

        for pcm_data in inputs:
            recognizer = KaldiRecognizer(self.vosk_model, 16000)
            recognizer.SetWords(False)

            chunk_size = 4000
            for i in range(0, len(pcm_data), chunk_size):
                chunk = pcm_data[i:i + chunk_size]
                recognizer.AcceptWaveform(chunk)
                
            result_json_str = recognizer.FinalResult()
            result_dict: Dict[str, Any] = json.loads(result_json_str)
            transcript: str = result_dict.get("text", "")
            transcriptions.append(transcript)
        return transcriptions

    def train(self, config_name: str, train_data: pd.DataFrame, val_data: pd.DataFrame) -> None:
        config_file_path: str = os.path.join(info.CONFIG_PATH, config_name)
        with open(config_file_path, "r", encoding="utf8") as f:
            config: Dict[str, Any] = json.load(f)

        output_directory: str = os.path.join(info.MODEL_PATH, self.name() + config["name"])

        # Vosk/Kaldi custom training usually involves compiling lexicon, language models (SRILM/KenLM),
        # and acoustic retraining. Here we provide the standard structured pipeline hook:
        print(f"Starting Vosk training/adaptation using config: {config['name']}")
        print(f"Training samples available: {len(train_data)}")
        if val_data is not None:
            print(f"Validation samples available: {len(val_data)}")

        # Placeholder for actual Kaldi/Vosk compilation scripts execution
        # After training completes, point model path to output directory
        self.is_trained = True
        self.model_path = output_directory
        # self.vosk_model = VoskNativeModel(output_directory)