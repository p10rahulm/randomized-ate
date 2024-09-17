from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from data_loaders.base_data_module import BaseDataModule


class IMDBDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item['text'], item['label']


class IMDBDataModule(BaseDataModule):
    def __init__(self, classification_word="Sentiment", val_split=0.1):
        self.classification_word = classification_word
        self.val_split = val_split
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.tokenizer = None
        self.load_data()

    def load_data(self):
        dataset = load_dataset("imdb")
        self.test_dataset = IMDBDataset(dataset['test'])

        train_val = dataset['train']
        val_size = int(len(train_val) * self.val_split)
        train_size = len(train_val) - val_size
        train_subset, val_subset = random_split(train_val, [train_size, val_size])
        self.train_dataset = IMDBDataset(train_subset)
        self.val_dataset = IMDBDataset(val_subset)

    def setup(self, tokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        texts, labels = zip(*batch)
        encodings = self.tokenizer(list(texts), truncation=True, padding=True, return_tensors="pt")
        return encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels)

    def get_dataloader(self, dataset, batch_size, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )

    def get_train_dataloader(self, batch_size):
        return self.get_dataloader(self.train_dataset, batch_size, shuffle=True)

    def get_val_dataloader(self, batch_size):
        return self.get_dataloader(self.val_dataset, batch_size)

    def get_test_dataloader(self, batch_size):
        return self.get_dataloader(self.test_dataset, batch_size)

    def get_full_train_dataloader(self, batch_size):
        full_train = torch.utils.data.ConcatDataset([self.train_dataset, self.val_dataset])
        return self.get_dataloader(full_train, batch_size, shuffle=True)

    def get_dataloaders(self, tokenizer, batch_size):
        self.setup(tokenizer)
        return self.get_train_dataloader(batch_size), self.get_val_dataloader(batch_size)

    def preprocess(self):
        train_data = [(item['text'], item['label']) for item in self.train_dataset.dataset]
        val_data = [(item['text'], item['label']) for item in self.val_dataset.dataset]
        return train_data + val_data

    def get_class_names(self):
        return ["negative", "positive"]