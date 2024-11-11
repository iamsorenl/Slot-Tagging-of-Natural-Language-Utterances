import torch
import numpy as np
import random
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import sys
import gensim.downloader as api

# Function to set a random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# define dataset
class POSDataset(Dataset):
    def __init__(self, tokens_list, tags_list=None, token_vocab=None, tag_vocab=None, training=True):
        # Tokenize the tokens and split tags if they are strings
        if isinstance(tokens_list[0], str):
            tokens_list = [tokens.split() for tokens in tokens_list]
        
        if tags_list is not None and isinstance(tags_list[0], str):
            tags_list = [tags.split() for tags in tags_list]
        else:
            tags_list = None

        # Create vocabularies if training and tags_list is available
        if training:
            self.token_vocab = {'<PAD>': 0, '<UNK>': 1}
            self.tag_vocab = {'<PAD>': 0}

            # Build vocab from training data
            for tokens in tokens_list:
                for token in tokens:
                    if token not in self.token_vocab:
                        self.token_vocab[token] = len(self.token_vocab)
            if tags_list is not None:
                for tags in tags_list:
                    for tag in tags:
                        if tag not in self.tag_vocab:
                            self.tag_vocab[tag] = len(self.tag_vocab)
        else:
            assert token_vocab is not None and tag_vocab is not None
            self.token_vocab = token_vocab
            self.tag_vocab = tag_vocab
        
        # Create inverse mapping for tags (common for both training and non-training scenarios)
        self.tag_vocab_inv = {idx: tag for tag, idx in self.tag_vocab.items()}

        # Convert tokens and tags to integer IDs during initialization
        self.corpus_token_ids = []
        self.corpus_tag_ids = []
        for i, tokens in enumerate(tokens_list):
            token_ids = [self.token_vocab.get(token, self.token_vocab['<UNK>']) for token in tokens]
            self.corpus_token_ids.append(torch.tensor(token_ids))
            if tags_list is not None:
                tag_ids = [self.tag_vocab[tag] for tag in tags_list[i]]
                self.corpus_tag_ids.append(torch.tensor(tag_ids))
            else:
                self.corpus_tag_ids.append(None)  # Placeholder for test set where tags are not available

    def __len__(self):
        return len(self.corpus_token_ids)

    def __getitem__(self, idx):
        return self.corpus_token_ids[idx], self.corpus_tag_ids[idx]
    
def collate_fn(batch):
    # Predefined padding value (assumed to be 0 for both tokens and tags)
    token_pad_value = 0
    tag_pad_value = 0

    # batch: [(token_ids, tag_ids), (token_ids, tag_ids), ...]
    # Separate tokens and tags
    token_ids = [item[0] for item in batch]
    tag_ids = [item[1] for item in batch if item[1] is not None]  # Exclude None for test set

    # Pad sequences
    sentences_padded = pad_sequence(token_ids, batch_first=True, padding_value=token_pad_value)
    if tag_ids:
        tags_padded = pad_sequence(tag_ids, batch_first=True, padding_value=tag_pad_value)
    else:
        tags_padded = None  # Handle the case for the test dataset

    return sentences_padded, tags_padded

def get_embedding_matrix(token_vocab, embedding_dim=100, glove_model_name="glove-wiki-gigaword-100"):
    # Load the GloVe model from gensim
    print(f"Loading GloVe model: {glove_model_name}")
    glove_model = api.load(glove_model_name)
    
    # Create the embedding matrix
    embedding_matrix = torch.zeros((len(token_vocab), embedding_dim))
    for word, idx in token_vocab.items():
        if word in glove_model:
            embedding_matrix[idx] = torch.tensor(glove_model[word])
        else:
            embedding_matrix[idx] = torch.randn(embedding_dim)  # Random initialization for unknown words
    return embedding_matrix

class SeqTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, num_layers, dropout, embedding_matrix=None):
        super().__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)  # Linear layer to compute attention scores
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)  # Fully connected layer for tag prediction

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)  # (batch_size, seq_len, embedding_dim)
        rnn_out, _ = self.bilstm(embeddings)  # (batch_size, seq_len, hidden_dim * 2)

        # Compute attention scores for each token's output in the sequence
        attn_scores = torch.tanh(self.attention(rnn_out))  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, seq_len, 1)

        # Apply the attention weights to the BiLSTM output
        weighted_rnn_out = attn_weights * rnn_out  # Element-wise multiplication (batch_size, seq_len, hidden_dim * 2)

        # Pass through the fully connected layer for per-token predictions
        outputs = self.fc(weighted_rnn_out)  # (batch_size, seq_len, tagset_size)

        return outputs

def main(train_file, test_file, output_file):
    # Set seed for reproducibility
    set_seed(10)

    # Load and split data into train and validation sets
    data = pd.read_csv(train_file)
    tokens_list = data['utterances'].tolist()
    tags_list = data['IOB Slot tags'].tolist()

    # Split data (80% training, 20% validation split)
    tokens_train, tokens_val, tags_train, tags_val = train_test_split(tokens_list, tags_list, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = POSDataset(tokens_train, tags_train, training=True)
    val_dataset = POSDataset(tokens_val, tags_val, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)
    test_dataset = POSDataset(pd.read_csv(test_file)['utterances'].tolist(), token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)

    # Display dataset size for verification
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Get embedding matrix
    embedding_matrix = get_embedding_matrix(train_dataset.token_vocab, embedding_dim=100)

    # Initialize model
    model = SeqTagger(
        vocab_size=len(train_dataset.token_vocab),
        tagset_size=len(train_dataset.tag_vocab),
        embedding_dim=100,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
        embedding_matrix=embedding_matrix
    )

    # Initialize loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.tag_vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    for epoch in range(40):
        # Training phase
        model.train()
        total_train_loss = 0
        for token_ids, tag_ids in train_loader:
            optimizer.zero_grad()
            outputs = model(token_ids)  # (batch_size, seq_len, tagset_size)

            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), tag_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation phase
        model.eval()
        total_val_loss = 0
        all_predictions = []
        all_tags = []

        with torch.no_grad():
            for token_ids, tag_ids in val_loader:
                outputs = model(token_ids)  # (batch_size, seq_len, tagset_size)
                outputs = outputs.view(-1, outputs.shape[-1])
                tag_ids = tag_ids.view(-1)

                loss = loss_fn(outputs, tag_ids)
                total_val_loss += loss.item()

                predictions = outputs.argmax(dim=1)
                mask = tag_ids != train_dataset.tag_vocab['<PAD>']  # Masking out padding tokens

                all_predictions.extend(predictions[mask].tolist())
                all_tags.extend(tag_ids[mask].tolist())

        # Compute train and validation loss
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)

        # Calculate F1 score
        f1 = f1_score(all_tags, all_predictions, average='weighted')

        print(f'Epoch {epoch + 1} | train_loss = {train_loss:.3f} | val_loss = {val_loss:.3f} | f1 = {f1:.3f}')

    # Testing phase
    print(f'\nTesting model on {test_file} and outputting to {output_file}\n')

    model.eval()
    predicted_tags = []

    with torch.no_grad():
        for token_ids, _ in test_loader:  # Note: test_loader only contains token_ids
            # Get model predictions
            outputs = model(token_ids)  # (batch_size, seq_len, tagset_size)
            predictions = outputs.argmax(dim=-1)  # Get the predicted indices for the tags

            # Convert predictions to tags using tag_vocab_inv (defined in POSDataset)
            for tokens, preds in zip(token_ids, predictions):
                # Convert predicted indices to tag strings
                predicted_seq = [train_dataset.tag_vocab_inv[idx.item()] for idx in preds]

                # Ensure the predicted sequence length matches the input sequence length
                input_length = tokens.ne(0).sum().item()  # Count non-padding tokens
                predicted_seq = predicted_seq[:input_length]  # Trim to match input length

                # Append to predictions as a space-separated string
                predicted_tags.append(' '.join(predicted_seq))

    # Save predictions to CSV
    df = pd.DataFrame({
        'ID': range(1, len(predicted_tags) + 1),  # Incremental IDs starting from 1
        'IOB Slot tags': predicted_tags
    })
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python run.py <train_file> <test_file> <output_file>")
        sys.exit(1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    main(train_file, test_file, output_file)
