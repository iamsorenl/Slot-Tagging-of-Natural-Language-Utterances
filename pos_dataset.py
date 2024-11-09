import torch
from torch.utils.data import Dataset
import pandas as pd

class POSDataset(Dataset):
    def __init__(self, dataframe, token_vocab=None, tag_vocab=None, training=True):
        """
        Initializes the dataset object for slot tagging of natural language utterances.
        Args:
            dataframe (pd.DataFrame): The dataframe containing the dataset with columns 'utterances' and 'IOB Slot tags'.
            training (bool): Flag indicating whether the object is being initialized in training mode.
            token_vocab (dict, optional): Predefined token vocabulary mapping tokens to indices. Required if not in training mode.
            tag_vocab (dict, optional): Predefined tag vocabulary mapping tags to indices. Required if not in training mode.
        Attributes:
            self.dataframe (pd.DataFrame): The dataframe containing the dataset.
            self.token_vocab (dict): The vocabulary mapping tokens to indices. Built from training data if in training mode.
            self.tag_vocab (dict): The vocabulary mapping tags to indices. Built from training data if in training mode.
            self.corpus_token_ids (list): List of PyTorch tensors containing token indices for each utterance.
            self.corpus_tag_ids (list): List of PyTorch tensors containing tag indices for each utterance.
        """
        # Store the dataframe
        self.dataframe = dataframe

        # Build mappings if in training mode
        if training:
            self.token_vocab = {'<PAD>': 0, '<UNK>': 1}
            self.tag_vocab = {'<PAD>': 0}

            # Build mappings from training data
            for _, row in self.dataframe.iterrows():
                tokens = row['utterances'].split()  # Tokenization step
                tags = row['IOB Slot tags'].split()  # Splitting IOB tags

                # Populate token_vocab
                for token in tokens:
                    if token not in self.token_vocab:
                        self.token_vocab[token] = len(self.token_vocab)

                # Populate tag_vocab
                for tag in tags:
                    if tag not in self.tag_vocab:
                        self.tag_vocab[tag] = len(self.tag_vocab)
            
            # Create inverse mappings for convenience
            self.tag_vocab_inv = {idx: tag for tag, idx in self.tag_vocab.items()}
        else:
            # Ensure mappings are provided during validation/testing
            assert token_vocab is not None and tag_vocab is not None
            self.token_vocab = token_vocab
            self.tag_vocab = tag_vocab

        # Preprocess and store the token and tag indices
        self.corpus_token_ids = []
        self.corpus_tag_ids = []

        for _, row in self.dataframe.iterrows():
            tokens = row['utterances'].split()  # Tokenize the utterance

            # Convert tokens to indices
            token_ids = [self.token_vocab.get(token, self.token_vocab['<UNK>']) for token in tokens]
            self.corpus_token_ids.append(torch.tensor(token_ids, dtype=torch.long))

            # Check if 'IOB Slot tags' exists; if so, process tags, else append None
            if 'IOB Slot tags' in row:
                tags = row['IOB Slot tags'].split()  # Split the IOB tags
                tag_ids = [self.tag_vocab.get(tag, self.tag_vocab['<PAD>']) for tag in tags]
                self.corpus_tag_ids.append(torch.tensor(tag_ids, dtype=torch.long))
            else:
                # If 'IOB Slot tags' column is not present (e.g., during testing), append None
                self.corpus_tag_ids.append(None)


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
