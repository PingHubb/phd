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

        # x = x + self.pe[:x.size(0), :]  # Dont use this, this is rubbish for our case
        # return self.dropout(x)


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


class _AI_DFM_AuxMLP(nn.Module):
    def __init__(self, aux_dim, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(aux_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)


class _AI_DFM_FrameEncoder(nn.Module):
    def __init__(self, in_channels, d_model, dropout):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, d_model),
        )

    def forward(self, x):
        return self.features(x)


class _AI_DFM_CNNTactileTransformerAux(nn.Module):
    def __init__(
        self,
        in_channels,
        aux_dim,
        seq_len,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        num_mode_classes,
        num_finger_classes,
        use_aux_features=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.use_aux_features = use_aux_features and aux_dim > 0
        self.frame_encoder = _AI_DFM_FrameEncoder(in_channels=in_channels, d_model=d_model, dropout=dropout)
        if self.use_aux_features:
            self.aux_encoder = _AI_DFM_AuxMLP(aux_dim=aux_dim, d_model=d_model, dropout=dropout)
            self.fusion = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.mode_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, num_mode_classes)
        )
        self.finger_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, num_finger_classes)
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 3)
        )

    def forward(self, x, aux=None):
        b, t, c, h, w = x.shape
        frame_tokens = self.frame_encoder(x.reshape(b * t, c, h, w)).reshape(b, t, -1)
        if self.use_aux_features and aux is not None:
            aux_tokens = self.aux_encoder(aux.reshape(b * t, -1)).reshape(b, t, -1)
            frame_tokens = self.fusion(torch.cat([frame_tokens, aux_tokens], dim=-1))
        frame_tokens = frame_tokens + self.pos_embedding[:, :t, :]
        device = frame_tokens.device
        causal_mask = torch.full((t, t), float("-inf"), device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        encoded = self.transformer(frame_tokens, mask=causal_mask)
        final_token = self.norm(encoded[:, -1, :])
        return {
            "mode_logits": self.mode_head(final_token),
            "finger_logits": self.finger_head(final_token),
            "velocity": self.velocity_head(final_token),
        }
