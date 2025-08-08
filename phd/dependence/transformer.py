import torch
from torch import nn
import math


# SENSOR_ROWS = 13  # 8 for arm, 13 for elbow
# SENSOR_COLS = 10  # 7 for arm, 10 for elbow
# NUM_SENSORS = SENSOR_ROWS * SENSOR_COLS


class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Note: In our case, x is (Batch, Seq_len, Features), so we permute
        x = x.permute(1, 0, 2)  # -> (Seq_len, Batch, Features)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1, 0, 2)  # -> (Batch, Seq_len, Features)
        return self.dropout(x)


class GestureBackbone(nn.Module):
    """
    The core CNN-Transformer model that learns the representations.
    It takes a sequence of sensor data and encodes it into rich feature vectors.
    """

    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, sensor_rows, sensor_cols):
        super().__init__()
        self.d_model = d_model

        self.height = sensor_rows
        self.width = sensor_cols

        # Part 1: CNN for Spatial Feature Extraction per time-step
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Automatically calculate the output size of the CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.height, self.width)
            cnn_output_size = self.conv(dummy_input).view(1, -1).size(1)

        # Part 2: Transformer for Temporal Analysis
        self.input_proj = nn.Linear(cnn_output_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

    def forward(self, src: torch.Tensor, src_padding_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = src.shape

        # Reshape for CNN: (Batch * Seq_len, Channels=1, Height, Width)
        src_reshaped = src.contiguous().view(-1, 1, self.height, self.width)

        # Pass through CNN and reshape back to a sequence
        cnn_out = self.conv(src_reshaped).view(batch_size, seq_len, -1)

        # Project CNN features to the Transformer's dimension (d_model)
        trans_input = self.input_proj(cnn_out) * math.sqrt(self.d_model)

        # Add positional encoding
        trans_input = self.pos_encoder(trans_input)

        # Feed into the transformer encoder
        return self.transformer_encoder(trans_input, src_key_padding_mask=src_padding_mask)


class HierarchicalGestureModel(nn.Module):
    def __init__(self, backbone, d_model, num_finger_classes, num_gesture_classes):
        super().__init__()
        self.backbone = backbone
        # Head 1: Predicts the number of fingers
        self.finger_classifier = nn.Linear(d_model, num_finger_classes)
        # Head 2: Predicts the gesture direction
        self.gesture_classifier = nn.Linear(d_model, num_gesture_classes)

    def forward(self, src, src_padding_mask):
        # Get the rich representation from the shared backbone
        representations = self.backbone(src, src_padding_mask)
        # Use the [CLS] token's representation for both classifications
        cls_representation = representations[:, 0, :]
        # Get predictions from both heads
        finger_logits = self.finger_classifier(cls_representation)
        gesture_logits = self.gesture_classifier(cls_representation)

        return finger_logits, gesture_logits


class ThreeLevelHierarchicalModel(nn.Module):
    """ A model with a shared backbone and three separate classification heads. """

    def __init__(self, backbone, d_model, num_finger_classes, num_gesture_classes, num_quality_classes):
        super().__init__()
        self.backbone = backbone
        self.finger_classifier = nn.Linear(d_model, num_finger_classes)
        self.gesture_classifier = nn.Linear(d_model, num_gesture_classes)
        self.quality_classifier = nn.Linear(d_model, num_quality_classes)

    def forward(self, src, src_padding_mask):
        representations = self.backbone(src, src_padding_mask)
        cls_representation = representations[:, 0, :]
        finger_logits = self.finger_classifier(cls_representation)
        gesture_logits = self.gesture_classifier(cls_representation)
        quality_logits = self.quality_classifier(cls_representation)
        return finger_logits, gesture_logits, quality_logits
