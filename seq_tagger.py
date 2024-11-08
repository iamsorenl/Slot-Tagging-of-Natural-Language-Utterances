import torch
import torch.nn as nn

class SeqTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(SeqTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)  # (batch_size, seq_len, embedding_dim)
        rnn_out, _ = self.lstm(embeddings)  # (batch_size, seq_len, hidden_dim)
        outputs = self.fc(rnn_out)  # (batch_size, seq_len, tagset_size)
        return outputs
