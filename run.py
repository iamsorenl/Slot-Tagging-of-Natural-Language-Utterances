import pandas as pd
import torch
import sys
from seq_tagger import SeqTagger
from pos_dataset import POSDataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

# Configuration Function
def get_config():
    return {
        "embedding_dim": 100,
        "hidden_dim": 128,
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 5
    }

def collate_fn(batch, train_dataset):
    token_ids = [item[0] for item in batch]
    tag_ids = [item[1] for item in batch]
    padding_value_token = train_dataset.word_to_index['<PAD>']
    padding_value_tag = train_dataset.tag_to_index['<PAD>']
    o_tag_index = train_dataset.tag_to_index.get('O', padding_value_tag)  # Use 'O' tag index if available, else fall back to padding

    # Counter to track how often 'O' tags are added
    o_tag_insertions = 0
    mismatch_cases = []  # To store mismatch details for further inspection

    # Handle length mismatches by adding 'O' tags if necessary
    for i in range(len(token_ids)):
        if len(token_ids[i]) != len(tag_ids[i]):
            # Log detailed information
            print(f"\n--- Warning: Length mismatch at index {i} ---")
            print(f"Tokens: {token_ids[i]}")
            print(f"Tags: {tag_ids[i]}")
            print(f"Token length: {len(token_ids[i])}, Tag length: {len(tag_ids[i])}")
            mismatch_cases.append((token_ids[i], tag_ids[i]))

            if len(token_ids[i]) > len(tag_ids[i]):
                mismatch_count = len(token_ids[i]) - len(tag_ids[i])
                # Convert tag_ids[i] to a list and extend with 'O' tags, then convert back to a tensor
                tag_ids[i] = torch.cat([tag_ids[i], torch.tensor([o_tag_index] * mismatch_count, dtype=torch.long)])
                o_tag_insertions += mismatch_count

    # Pad sequences with `<PAD>` token
    utterances_padded = pad_sequence(token_ids, batch_first=True, padding_value=padding_value_token)
    tags_padded = pad_sequence(tag_ids, batch_first=True, padding_value=padding_value_tag)

    # Report the number of 'O' tag insertions
    if o_tag_insertions > 0:
        print(f"\nTotal 'O' tag insertions in this batch: {o_tag_insertions}")

    # Optional: Save mismatch cases to inspect later if necessary
    # This could be written to a file or used in further analysis.
    if mismatch_cases:
        with open('mismatch_debug_log.txt', 'a') as log_file:
            for tokens, tags in mismatch_cases:
                log_file.write(f"Tokens: {tokens}\n")
                log_file.write(f"Tags: {tags}\n")
                log_file.write(f"Token length: {len(tokens)}, Tag length: {len(tags)}\n")
                log_file.write("\n")

    return utterances_padded, tags_padded

# Function to Split Data
def split_data(train_data, val_size=0.2):
    return train_test_split(train_data, test_size=val_size, random_state=12, shuffle=True)

# Function to Create Data Loaders
def get_data_loaders(train_dataset, val_dataset, batch_size, collate_fn):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, train_dataset))
    return train_loader, val_loader

# Function to Initialize Model
def initialize_model(vocab_size, tagset_size, embedding_dim, hidden_dim):
    model = SeqTagger(vocab_size, tagset_size, embedding_dim, hidden_dim)
    return model

def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, train_dataset):
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        for token_ids, tag_ids in train_loader:
            optimizer.zero_grad()
            outputs = model(token_ids)  # (batch_size, seq_len, tagset_size)

            # Debugging: Print shapes
            # print(f"outputs shape: {outputs.shape}, tag_ids shape: {tag_ids.shape}")

            # Reshape outputs and tag_ids for loss computation
            batch_size, seq_len, tagset_size = outputs.shape
            outputs = outputs.view(batch_size * seq_len, tagset_size)
            tag_ids = tag_ids.view(-1)

            # Check if shapes match
            assert outputs.shape[0] == tag_ids.shape[0], f"Shape mismatch: outputs {outputs.shape[0]} vs tag_ids {tag_ids.shape[0]}"

            # Compute the loss
            loss = loss_fn(outputs, tag_ids)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation Phase
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
                mask = tag_ids != train_dataset.tag_to_index['<PAD>']
                all_predictions.extend(predictions[mask].tolist())
                all_tags.extend(tag_ids[mask].tolist())

        # Compute Metrics
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)
        f1 = f1_score(all_tags, all_predictions, average='weighted')

        print(f'Epoch {epoch+1} | train_loss = {train_loss:.3f} | val_loss = {val_loss:.3f} | f1 = {f1:.3f}')

# Main Function
def main(training_file, testing_file, output_file):
    config = get_config()

    # Load and split data
    train_df = pd.read_csv(training_file)
    train_df, val_df = split_data(train_df, val_size=0.2)

    # Create datasets
    train_dataset = POSDataset(train_df, training=True)
    val_dataset = POSDataset(val_df, word_to_index=train_dataset.word_to_index, tag_to_index=train_dataset.tag_to_index, training=False)

    # Create data loaders
    train_loader, val_loader = get_data_loaders(train_dataset, val_dataset, config["batch_size"], collate_fn)

    # Initialize model
    model = initialize_model(
        vocab_size=len(train_dataset.word_to_index),
        tagset_size=len(train_dataset.tag_to_index),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"]
    )

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.tag_to_index['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Train the model
    train_model(model, train_loader, val_loader, loss_fn, optimizer, config["num_epochs"], train_dataset)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run.py <train_data> <test_data> <output_file>")
        sys.exit(1)
    
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3]
    
    main(arg1, arg2, arg3)
