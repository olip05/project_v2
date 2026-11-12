"""Vosk model implementation with resampling and parallel inference"""
from typing import List, Any, Dict
import os
import json
import concurrent.futures
import numpy as np
import librosa
from pandas import DataFrame
from vosk import Model as VoskNativeModel, KaldiRecognizer # type: ignore
from model import Model
import base_info as info

class VoskModel(Model):
    """Vosk model wrapper implementing the Model interface"""

    def __init__(self, model_name: str) -> None:
        super().__init__(model_name=model_name)
        self.is_trained: bool = True  # Vosk is pretrained out-of-the-box
        self.vosk_model = VoskNativeModel(os.path.join(info.MODEL_PATH, model_name))

    def format_inputs(self, inputs: List[Any]) -> List[bytes]:
        """
        Resample raw audio from original 24kHz to 16kHz (required by Vosk)
        and convert to 16-bit PCM bytes.
        """
        pcm_bytes_list: List[bytes] = []
        for audio in inputs:
            audio_np = np.asarray(audio, dtype=np.float32)

            if info.SR != 16000:
                audio_np = librosa.resample(audio_np, orig_sr=info.SR, target_sr=16000)

            audio_int16 = (audio_np * 32767).astype(np.int16)
            pcm_bytes_list.append(audio_int16.tobytes())

        return pcm_bytes_list

    def load(self, weight_path: str) -> None:
        """Load Vosk model from path."""
        self.vosk_model = VoskNativeModel(os.path.join(info.MODEL_PATH, weight_path))
        self.is_trained = True

    def _process_single_audio(self, pcm_data: bytes) -> str:
        """Process a single PCM byte stream through KaldiRecognizer."""
        recognizer = KaldiRecognizer(self.vosk_model, 16000)
        recognizer.SetWords(False) # type: ignore

        chunk_size = 4000
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i:i + chunk_size]
            recognizer.AcceptWaveform(chunk) # type: ignore

        result_json_str = recognizer.FinalResult() # type: ignore
        result_dict: Dict[str, Any] = json.loads(result_json_str) # type: ignore
        return str(result_dict.get("text", ""))

    def predict(self, inputs: Any) -> List[str]:
        """
        Run parallelized inference over inputs using multi-threading 
        to maximize CPU efficiency since Vosk is CPU-bound.
        """
        max_workers = min(4, os.cpu_count() or 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            transcriptions = list(executor.map(self._process_single_audio, inputs))

        return transcriptions

    def train(self, config_name: str, train_data: DataFrame, val_data: DataFrame) -> None:
        pass
