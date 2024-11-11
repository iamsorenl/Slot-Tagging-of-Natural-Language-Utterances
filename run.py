import torch
import numpy as np
import random
from sklearn.metrics import f1_score as sklearn_f1_score
from seqeval.metrics import f1_score as seqeval_f1_score, classification_report
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

# Define dataset class
class POSDataset(Dataset):
    def __init__(self, tokens_list, tags_list=None, token_vocab=None, tag_vocab=None, training=True):
        # Tokenize and create vocabularies...
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
                self.corpus_tag_ids.append(None)

    def __len__(self):
        return len(self.corpus_token_ids)

    def __getitem__(self, idx):
        return self.corpus_token_ids[idx], self.corpus_tag_ids[idx]

# Custom collate function to pad sequences
def collate_fn(batch):
    # Predefined padding value (assumed to be 0 for both tokens and tags)
    token_pad_value = 0
    tag_pad_value = 0

    # batch: [(token_ids, tag_ids), (token_ids, tag_ids), ...]
    # Separate tokens and tags
    token_ids = [item[0] for item in batch]
    tag_ids = [item[1] for item in batch if item[1] is not None]
    sentences_padded = pad_sequence(token_ids, batch_first=True, padding_value=token_pad_value)
    if tag_ids:
        tags_padded = pad_sequence(tag_ids, batch_first=True, padding_value=tag_pad_value)
    else:
        tags_padded = None
    return sentences_padded, tags_padded

# Function to get the embedding matrix
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
            embedding_matrix[idx] = torch.randn(embedding_dim)
    return embedding_matrix

# Define the BiLSTM model with attention
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
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)
        rnn_out, _ = self.bilstm(embeddings)
        attn_scores = torch.tanh(self.attention(rnn_out))
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_rnn_out = attn_weights * rnn_out
        outputs = self.fc(weighted_rnn_out)
        return outputs

# Helper function to convert tags from custom B_ or I_ format to B- or I- format
def convert_tags_to_iob2(tags):
    """Convert tags from custom B_ or I_ format to B- or I- format."""
    converted_tags = []
    for tag in tags:
        if tag.startswith('B_') or tag.startswith('I_'):
            converted_tags.append(tag.replace('_', '-'))
        else:
            converted_tags.append(tag)  # Keep 'O' or any other tag as-is
    return converted_tags

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
            outputs = model(token_ids)
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
                outputs = model(token_ids)
                outputs = outputs.view(-1, outputs.shape[-1])
                tag_ids = tag_ids.view(-1)

                loss = loss_fn(outputs, tag_ids)
                total_val_loss += loss.item()

                predictions = outputs.argmax(dim=1)
                mask = tag_ids != train_dataset.tag_vocab['<PAD>']
                all_predictions.extend(predictions[mask].tolist())
                all_tags.extend(tag_ids[mask].tolist())

        # Convert indices back to labels
        all_predictions_labels = [train_dataset.tag_vocab_inv[idx] for idx in all_predictions]
        all_tags_labels = [train_dataset.tag_vocab_inv[idx] for idx in all_tags]

        # Function to convert tags from custom format to IOB2 standard for seqeval
        def convert_tags_to_iob2(tags):
            converted_tags = []
            for tag in tags:
                if tag.startswith('B_') or tag.startswith('I_'):
                    converted_tags.append(tag.replace('_', '-'))
                else:
                    converted_tags.append(tag)
            return converted_tags

        # Apply conversion
        all_predictions_labels = convert_tags_to_iob2(all_predictions_labels)
        all_tags_labels = convert_tags_to_iob2(all_tags_labels)

        # Group tags into sequences for seqeval
        grouped_predictions = []
        grouped_tags = []
        current_pred = []
        current_tag = []
        for i, (pred, tag) in enumerate(zip(all_predictions_labels, all_tags_labels)):
            if pred != '<PAD>':
                current_pred.append(pred)
                current_tag.append(tag)
            if i == len(all_predictions_labels) - 1 or all_predictions_labels[i + 1] == '<PAD>':
                grouped_predictions.append(current_pred)
                grouped_tags.append(current_tag)
                current_pred = []
                current_tag = []

        # Calculate F1 scores using both sklearn and seqeval
        f1_sklearn = sklearn_f1_score(all_tags, all_predictions, average='weighted')
        f1_seqeval = seqeval_f1_score(grouped_tags, grouped_predictions)
        print(f'Epoch {epoch + 1} | train_loss = {total_train_loss / len(train_loader):.3f} | val_loss = {total_val_loss / len(val_loader):.3f} | sklearn F1 = {f1_sklearn:.3f} | seqeval F1 = {f1_seqeval:.3f}')

    model.eval()
    predicted_tags = []

    with torch.no_grad():
        for token_ids, _ in test_loader:
            outputs = model(token_ids)
            predictions = outputs.argmax(dim=-1)
            for tokens, preds in zip(token_ids, predictions):
                # Convert predicted indices to tag strings
                predicted_seq = [train_dataset.tag_vocab_inv[idx.item()] for idx in preds]
                input_length = tokens.ne(0).sum().item()
                predicted_seq = predicted_seq[:input_length]
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

