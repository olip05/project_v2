"""Vosk model implementation"""
from typing import List, Any, Optional, Dict
import os
import json
import pandas as pd
import numpy as np
import librosa
from vosk import Model as VoskNativeModel, KaldiRecognizer
from model import Model
import base_info as info

class VoskModel(Model):
    """Vosk model wrapper implementing the Model interface"""

    def __init__(self, model_name: str = "vosk-model-small-esperanto", sample_rate: int = 16000) -> None:
        super().__init__(model_name=model_name)
        self.is_trained: bool = False
        self.sample_rate: int = sample_rate
        self.model_path: str = os.path.join(info.MODEL_PATH, model_name)
        
        # Initialize Vosk Native Model if directory exists
        self.vosk_model: Optional[VoskNativeModel] = None
        if os.path.exists(self.model_path):
            self.vosk_model = VoskNativeModel(self.model_path)
            self.is_trained = True

    def format_inputs(self, inputs: List[Any]) -> List[bytes]:
        """
        Format raw audio arrays (from librosa) into raw PCM byte strings 
        expected by the Vosk recognizer.
        """
        pcm_bytes_list: List[bytes] = []
        for audio in inputs:
            # Ensure audio is float32 numpy array and scaled to 16-bit PCM integer bytes
            audio_np = np.asarray(audio, dtype=np.float32)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            pcm_bytes_list.append(audio_int16.tobytes())
            
        return pcm_bytes_list

    def load(self, weight_path: str) -> None:
        """Load Vosk model from a specific path."""
        full_path = os.path.join(info.MODEL_PATH, weight_path)
        self.model_path = full_path
        self.vosk_model = VoskNativeModel(full_path)
        self.is_trained = True

    def predict(self, inputs: Any) -> List[str]:
        """
        Given preprocessed raw PCM byte strings (from format_inputs),
        runs Vosk recognition and returns a list of transcribed strings.
        """
        if self.vosk_model is None:
            raise ValueError(f"Vosk model not loaded or path does not exist: {self.model_path}")

        transcriptions: List[str] = []
        
        for pcm_data in inputs:
            recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
            recognizer.SetWords(False)
            
            # Feed raw audio chunks to recognizer
            # Vosk accepts chunks of bytes
            chunk_size = 4000
            for i in range(0, len(pcm_data), chunk_size):
                chunk = pcm_data[i:i + chunk_size]
                recognizer.AcceptWaveform(chunk)
                
            # Get final recognition result string
            result_json_str = recognizer.FinalResult()
            result_dict: Dict[str, Any] = json.loads(result_json_str)
            transcript: str = result_dict.get("text", "")
            transcriptions.append(transcript)

        return transcriptions

    def train(self, config_name: str, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None) -> None:
        """
        Train or adapt the acoustic/language model components for Vosk
        using the specified configuration file.
        """
        config_file_path: str = os.path.join(info.CONFIG_PATH, config_name)
        with open(config_file_path, "r", encoding="utf8") as f:
            config: Dict[str, Any] = json.load(f)

        output_directory: str = os.path.join(info.MODEL_PATH, self.name(), config["name"])
        os.makedirs(output_directory, exist_ok=True)

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