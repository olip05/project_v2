"""HMM-DNN model implementation"""
from typing import List, Any, Optional, Dict
import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import librosa
from model import Model
import base_info as info

class AcousticDNN(nn.Module):
    """A simple Feedforward/CNN Acoustic DNN for the HMM-DNN pipeline."""
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class HMMDNNModel(Model):
    """HMM-DNN model implementation based on the Model interface."""

    def __init__(self, model_name: str, input_dim: int = 13, num_classes: int = 40) -> None:
        super().__init__(model_name=model_name)
        self.is_trained: bool = False
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim: int = input_dim
        self.num_classes: int = num_classes
        
        self.acoustic_model: AcousticDNN = AcousticDNN(input_dim=input_dim, num_classes=num_classes).to(self.device)

    def format_inputs(self, inputs: List[Any]) -> List[np.ndarray]:
        mfcc_features: List[np.ndarray] = []
        for audio in inputs:
            mfccs = librosa.feature.mfcc(y=np.asarray(audio, dtype=np.float32), sr=info.SR, n_mfcc=self.input_dim)
            mfcc_features.append(mfccs.T)            
        return mfcc_features

    def load(self, weight_path: str) -> None:
        checkpoint = torch.load(os.path.join(info.MODEL_PATH, weight_path), map_location=self.device)
        self.acoustic_model.load_state_dict(checkpoint)
        self.acoustic_model.eval()
        self.is_trained = True

    def predict(self, inputs: Any) -> List[str]:
        self.acoustic_model.eval()
        transcriptions: List[str] = []

        with torch.no_grad():
            for mfccs in inputs:
                mfcc_tensor = torch.tensor(mfccs, dtype=torch.float32).to(self.device)
                
                # Forward pass through acoustic model -> frame log-likelihoods / posteriors
                outputs = self.acoustic_model(mfcc_tensor)
                predicted_classes = torch.argmax(outputs, dim=-1).cpu().numpy()
                
                # TODO: Hier greift normalerweise der HMM-Viterbi-Dekoder mit einem N-Gramm Sprachmodell.
                # Als erster Prototyp / Baseline können wir die Klassen in Text mappen oder einen 
                # Greedy Decoder verwenden:
                dummy_text = f"decoding_result_{len(predicted_classes)}_frames"
                transcriptions.append(dummy_text)

        return transcriptions

    def train(self, config_name: str, train_data: pd.DataFrame, val_data: Optional[pd.DataFrame] = None) -> None:
        config_file_path: str = os.path.join(info.CONFIG_PATH, config_name)
        with open(config_file_path, "r", encoding="utf8") as f:
            config: Dict[str, Any] = json.load(f)

        output_directory: str = os.path.join(info.MODEL_PATH, self.name(), config["name"])
        os.makedirs(output_directory, exist_ok=True)

        print(f"Training HMM-DNN Acoustic Model with config: {config['name']}")
        print(f"Epochs: {config['epochs']}, Batch Size: {config['batch_size']}, LR: {config['lr']}")

        # Hier implementierst du dein Trainings-Setup (Loss, Optimizer, DataLoader für MFCCs)
        # ...
        
        self.is_trained = True
        torch.save(self.acoustic_model.state_dict(), os.path.join(output_directory, "acoustic_model.pt"))