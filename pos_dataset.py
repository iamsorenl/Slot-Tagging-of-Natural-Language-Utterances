import torch
from torch.utils.data import Dataset
import pandas as pd

class POSDataset(Dataset):
    def __init__(self, dataframe, word_to_index=None, tag_to_index=None, max_seq_length=None, training=True):
        """
        Initialize the dataset with a DataFrame, word-to-index mapping, tag-to-index mapping, 
        and an optional maximum sequence length for padding/truncation.
        """
        self.dataframe = dataframe
        self.max_seq_length = max_seq_length

        # Build mappings if in training mode
        if training:
            self.word_to_index = {'<PAD>': 0, '<UNK>': 1}
            self.tag_to_index = {'<PAD>': 0}

            # Build mappings from training data
            for _, row in self.dataframe.iterrows():
                tokens = row['utterances'].split()  # Tokenization step
                tags = row['IOB Slot tags'].split()  # Splitting IOB tags

                # Populate word_to_index
                for token in tokens:
                    if token not in self.word_to_index:
                        self.word_to_index[token] = len(self.word_to_index)

                # Populate tag_to_index
                for tag in tags:
                    if tag not in self.tag_to_index:
                        self.tag_to_index[tag] = len(self.tag_to_index)
        else:
            assert word_to_index is not None and tag_to_index is not None, "word_to_index and tag_to_index must be provided during validation/testing"
            self.word_to_index = word_to_index
            self.tag_to_index = tag_to_index

        # Preprocess and store the token and tag indices
        self.corpus_token_ids = []
        self.corpus_tag_ids = []

        for _, row in self.dataframe.iterrows():
            tokens = row['utterances'].split()  # Tokenize the utterance
            tags = row['IOB Slot tags'].split()  # Split the IOB tags

            # Convert tokens and tags to indices
            token_ids = [self.word_to_index.get(token, self.word_to_index['<UNK>']) for token in tokens]
            tag_ids = [self.tag_to_index.get(tag, self.tag_to_index['<PAD>']) for tag in tags]

            # Handle padding/truncation if max_seq_length is specified
            if self.max_seq_length:
                token_ids = token_ids[:self.max_seq_length] + [self.word_to_index['<PAD>']] * max(0, self.max_seq_length - len(token_ids))
                tag_ids = tag_ids[:self.max_seq_length] + [self.tag_to_index['<PAD>']] * max(0, self.max_seq_length - len(tag_ids))

            self.corpus_token_ids.append(torch.tensor(token_ids, dtype=torch.long))
            self.corpus_tag_ids.append(torch.tensor(tag_ids, dtype=torch.long))

    def __len__(self):
        """
        Return the number of examples (rows) in the dataset.
        """
        return len(self.corpus_token_ids)

    def __getitem__(self, idx):
        """
        Retrieve and process the tokenized sentence and its corresponding IOB tags at index 'idx'.
        Return the processed token indices and tag indices as PyTorch tensors.
        """
        return self.corpus_token_ids[idx], self.corpus_tag_ids[idx]