import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        # LSTM 레이어에 Dropout 추가
        self.lstm = nn.LSTM(input_size, hidden_layer_size,
                            num_layers=2, batch_first=True,
                            bidirectional=False, dropout=dropout)

        # Fully Connected 레이어
        self.fc1 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.tanh1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_layer_size, hidden_layer_size // 2)
        self.tanh2 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_layer_size // 2, output_size)

    def forward(self, x):
        # LSTM 출력: (batch, seq_len, hidden_dim)
        x, _ = self.lstm(x)
        # 마지막 시점의 출력만 사용
        x = x[:, -1, :]  # (batch, hidden_dim)

        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.tanh2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x
