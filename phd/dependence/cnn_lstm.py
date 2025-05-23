from torch import nn, optim
import torch


class GestureCNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim, num_layers, dropout_rate):
        super(GestureCNNLSTM, self).__init__()
        # Adjust the height and width based on input_size
        self.height = 7  # 7
        self.width = 8  # 8
        if self.height * self.width != input_size:
            raise ValueError(f'Input size {input_size} does not match expected grid size {self.height}x{self.width}')
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        conv_output_size = self._get_conv_output_size()
        self.lstm = nn.LSTM(conv_output_size, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def _get_conv_output_size(self):
        dummy_input = torch.zeros(1, 1, self.height, self.width)
        output = self.conv(dummy_input)
        output_size = output.view(1, -1).size(1)
        return output_size

    def forward(self, x):
        # Unpack sequences
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        batch_size = x.size(0)
        seq_len = x.size(1)
        # Reshape to (batch_size * seq_len, 1, H, W)
        x = x.contiguous().view(-1, 1, self.height, self.width)
        x = self.conv(x)
        x = x.view(batch_size, seq_len, -1)
        # Pack sequences again
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(x)
        logits = self.classifier(hidden[-1])
        return logits
