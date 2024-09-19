import multiprocessing
from functools import partial

import torch
from torch.utils.data import DataLoader
from data_loaders.imdb import IMDBDataModule
import random
from tqdm import tqdm
from data_loaders.simple_dataloader import SimpleDataset

class ATEDataModule(IMDBDataModule):
    def __init__(self, classification_word, model, perturbation_rate=0.5, num_perturbations=25, n_gram_length=5, change_threshold=0.5):
        super().__init__(classification_word)
        self.model = model
        self.perturbation_rate = perturbation_rate
        self.num_perturbations = num_perturbations
        self.n_gram_length = n_gram_length
        self.change_threshold = change_threshold
        self.ate_train_data = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = model.tokenizer

    def prepare_data(self, batch_size=32):
        super().load_data()  # Load the IMDB data
        self._prepare_ate_data(batch_size=batch_size)

    def _perturb_text(self, text, vocab):
        tokens = text.split()
        total_tokens = len(tokens)
        num_to_perturb = max(3, int(total_tokens * self.perturbation_rate))
        num_ngrams = num_to_perturb // self.n_gram_length + 1

        perturbed_versions = []
        for _ in range(self.num_perturbations):
            sentence_as_tokens = tokens[:]
            perturbed_tokens = []
            perturbed_indices = []

            for _ in range(num_ngrams):
                start_idx = random.randint(0, total_tokens - self.n_gram_length)
                perturbed_indices.extend(range(start_idx, start_idx + self.n_gram_length))

            perturbed_indices = sorted(set(perturbed_indices))

            for idx in perturbed_indices:
                old_token = sentence_as_tokens[idx]
                new_token = random.choice(list(vocab.keys()))
                sentence_as_tokens[idx] = new_token
                perturbed_tokens.append(old_token)

            perturbed_part = ' '.join(perturbed_tokens)
            perturbed_text = ' '.join(sentence_as_tokens)
            perturbed_versions.append((perturbed_text, perturbed_part))

        return perturbed_versions
    

    def _pre_compute_perturbations(self, batch_size=32):
        train_data = self.preprocess()  # Use the preprocess method from IMDBDataModule
        vocab = self.get_vocab()
        all_perturbations = []

        for i in tqdm(range(0, len(train_data), batch_size), desc="Pre-computing perturbations"):
            batch = train_data[i:i+batch_size]
            texts, labels = zip(*batch)

            # Process the batch of texts in parallel
            perturbed_batch = [self._perturb_text(text, vocab) for text in texts]

            # Flatten and extend the results
            for text, label, perturbed_versions in zip(texts, labels, perturbed_batch):
                all_perturbations.extend([
                    (text, perturbed, perturbed_part, label)
                    for perturbed, perturbed_part in perturbed_versions
                ])

        return all_perturbations

    def _prepare_ate_data(self, batch_size=32):
        print("Pre-computing perturbations...")
        all_perturbations = self._pre_compute_perturbations(batch_size)
        
        print(f"Total perturbations generated: {len(all_perturbations)}")
        
        ate_data = []
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for i in tqdm(range(0, len(all_perturbations), batch_size), desc="Scoring perturbations"):
                batch = all_perturbations[i:i+batch_size]
                original_inputs = [item[0] for item in batch]
                perturbed_inputs = [item[1] for item in batch]
                
                original_encodings = self.tokenizer(original_inputs, truncation=True, padding=True, return_tensors="pt")
                perturbed_encodings = self.tokenizer(perturbed_inputs, truncation=True, padding=True, return_tensors="pt")
                
                original_scores = self.model(**{k: v.to(self.device) for k, v in original_encodings.items()})
                perturbed_scores = self.model(**{k: v.to(self.device) for k, v in perturbed_encodings.items()})
                
                if hasattr(original_scores, 'logits'):
                    original_scores = original_scores.logits
                    perturbed_scores = perturbed_scores.logits
                
                score_differences = torch.abs(original_scores - perturbed_scores)
                max_changes = torch.max(score_differences, dim=1)[0]
                
                for (_, _, perturbed_part, label), max_change in zip(batch, max_changes):
                    if max_change >= self.change_threshold:
                        ate_data.append((perturbed_part, label))

        self.ate_train_data = ate_data
        print(f"Generated {len(ate_data)} ATE training samples.")

    def get_ate_train_dataloader(self, batch_size=32):
        if self.ate_train_data is None:
            raise ValueError("ATE data has not been prepared. Call prepare_data() first.")
        
        dataset = SimpleDataset(self.ate_train_data, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

    def get_dataloaders(self, batch_size):
        return self.get_ate_train_dataloader(batch_size), self.get_val_dataloader(batch_size)
    
    def get_full_train_dataloader(self, batch_size=32):
        if self.ate_train_data is None:
            raise ValueError("ATE data has not been prepared. Call prepare_data() first.")
        
        # Combine train and validation data
        full_train_data = self.ate_train_data + [
            (text, label) for text, label in self.preprocess()[len(self.train_dataset):]
        ]
        
        dataset = SimpleDataset(full_train_data, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)

    def get_vocab(self):
        return self.tokenizer.get_vocab()