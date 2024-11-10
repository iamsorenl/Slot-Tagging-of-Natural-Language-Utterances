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

# collate token_ids and tag_ids to make mini-batches
def collate_fn(batch, train_dataset):
    # Separate sentences and tags
    token_ids = [item[0] for item in batch]
    tag_ids = [item[1] for item in batch if item[1] is not None]  # Only keep non-None tag_ids

    # Pad sequences for token_ids
    sentences_padded = pad_sequence(token_ids, batch_first=True, padding_value=train_dataset.token_vocab['<PAD>'])
    
    # Handle tags padding only if tag_ids are present
    if tag_ids:
        tags_padded = pad_sequence(tag_ids, batch_first=True, padding_value=train_dataset.tag_vocab['<PAD>'])
    else:
        tags_padded = None  # No tags for test data

    return sentences_padded, tags_padded

# Function to Split Data
def split_data(train_data, val_size=0.2):
    return train_test_split(train_data, test_size=val_size, random_state=12, shuffle=True)

# Main Function
def main(training_file, testing_file, output_file):
   # Load and split data
    train_df = pd.read_csv(training_file)
    train_df, val_df = split_data(train_df, val_size=0.2)

    # Create datasets
    train_dataset = POSDataset(train_df, training=True)
    val_dataset = POSDataset(val_df, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)

    # Create dataloaders
    embedding_dim = 100
    hidden_dim = 128
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 5

    # Initialize model
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, train_dataset))

    # Initialize model
    model = SeqTagger(
    vocab_size=len(train_dataset.token_vocab),
    tagset_size=len(train_dataset.tag_vocab),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
    )

    # Initialize loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=train_dataset.tag_vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        for token_ids, tag_ids in train_loader:
            optimizer.zero_grad()

            outputs = model(token_ids)  # (batch_size, seq_len, tagset_size)

            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), tag_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation
        model.eval()
        total_val_loss = 0
        all_predictions = []
        all_tags = []

        with torch.no_grad():
            outputs = model(token_ids)  # (batch_size, seq_len, tagset_size)

            outputs = outputs.view(-1, outputs.shape[-1])
            tag_ids = tag_ids.view(-1)
            loss = loss_fn(outputs, tag_ids)
            total_val_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            mask = tag_ids != train_dataset.tag_vocab['<PAD>']

            all_predictions.extend(predictions[mask].tolist())
            all_tags.extend(tag_ids[mask].tolist())

        # compute train and val loss
        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / len(val_loader)

        # Calculate F1 score
        f1 = f1_score(all_tags, all_predictions, average='weighted')

        print(f'{epoch = } | train_loss = {train_loss:.3f} | val_loss = {val_loss:.3f} | f1 = {f1:.3f}')

        # Testing phase
        test_df = pd.read_csv(testing_file)
        test_dataset = POSDataset(test_df, token_vocab=train_dataset.token_vocab, tag_vocab=train_dataset.tag_vocab, training=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, train_dataset))

        model.eval()
        test_predictions = []
        with torch.no_grad():
            for batch in test_loader:
                # Extract token_ids
                if isinstance(batch, tuple):
                    token_ids = batch[0]  # Ignore tag_ids since they will be None
                else:
                    token_ids = batch

                outputs = model(token_ids)
                predictions = outputs.argmax(dim=2)  # Get predicted tag indices (batch_size, seq_len)

                # Convert predictions to tag names using tag_vocab (inverse mapping)
                for i, pred_seq in enumerate(predictions):
                    # Get the length of the input sequence (non-padding tokens)
                    input_length = (token_ids[i] != train_dataset.token_vocab['<PAD>']).sum().item()
                    
                    # Trim predictions to match input length
                    pred_tags = [train_dataset.tag_vocab_inv[idx] for idx in pred_seq.tolist()[:input_length]]
                    test_predictions.append(' '.join(pred_tags))

        # Save predictions with IDs to the output file
        with open(output_file, 'w') as f:
            f.write("ID,IOB Slot tags\n")
            for idx, tags in zip(test_df['ID'], test_predictions):
                f.write(f"{idx},{tags}\n")



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run.py <train_data> <test_data> <output_file>")
        sys.exit(1)
    
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    arg3 = sys.argv[3]
    
    main(arg1, arg2, arg3)
