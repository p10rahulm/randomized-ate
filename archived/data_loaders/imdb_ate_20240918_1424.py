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
        self.tokenizer = model.tokenizer  # Assuming the model has a tokenizer attribute


    def prepare_data(self, batch_size=32):
        # Load the IMDB data
        self.load_data()
        # Prepare the ATE data
        self._prepare_ate_data(batch_size=batch_size)

    def _perturb_batch(self, batch, vocab):
        print(len(batch))
        original_inputs, original_attention_mask, original_labels = batch
        perturbed_data = []

        for original_input, original_label in zip(original_inputs, original_labels):
            tokens = original_input.split()
            total_tokens = len(tokens)
            num_to_perturb = max(3, int(total_tokens * self.perturbation_rate))
            num_ngrams = num_to_perturb // self.n_gram_length + 1

            batch_perturbed = []
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

                perturbed_part_of_input = ' '.join(perturbed_tokens)
                input_after_perturbation = ' '.join(sentence_as_tokens)
                batch_perturbed.append((original_input, input_after_perturbation, perturbed_part_of_input, original_label))
            
            perturbed_data.append(batch_perturbed)

        return perturbed_data
    
    def _perturb_batch(self, batch, vocab):
        input_ids, attention_mask, original_labels = batch
        perturbed_data = []

        # Convert input_ids back to text
        original_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        for original_input, original_label in zip(original_inputs, original_labels):
            tokens = original_input.split()
            total_tokens = len(tokens)
            num_to_perturb = max(3, int(total_tokens * self.perturbation_rate))
            num_ngrams = num_to_perturb // self.n_gram_length + 1

            batch_perturbed = []
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

                perturbed_part_of_input = ' '.join(perturbed_tokens)
                input_after_perturbation = ' '.join(sentence_as_tokens)
                batch_perturbed.append((original_input, input_after_perturbation, perturbed_part_of_input, original_label))
            
            perturbed_data.append(batch_perturbed)

        return perturbed_data

    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    # def _prepare_ate_data(self):
    #     train_loader = self.get_train_dataloader(batch_size=32)
    #     vocab = self.get_vocab()
    #     ate_data = []

    #     self.model.eval()
    #     self.model.to(self.device)

    #     with torch.no_grad():
    #         for batch in tqdm(train_loader, desc="Preparing ATE data"):
    #             perturbed_batches = self._perturb_batch(batch, vocab)
                
    #             for perturbed_batch in perturbed_batches:
    #                 original_inputs = [item[0] for item in perturbed_batch]
    #                 perturbed_inputs = [item[1] for item in perturbed_batch]
                    
    #                 original_encodings = self.tokenizer(original_inputs, truncation=True, padding=True, return_tensors="pt")
    #                 perturbed_encodings = self.tokenizer(perturbed_inputs, truncation=True, padding=True, return_tensors="pt")
                    
    #                 original_scores = self.model(**{k: v.to(self.device) for k, v in original_encodings.items()}).logits
    #                 perturbed_scores = self.model(**{k: v.to(self.device) for k, v in perturbed_encodings.items()}).logits
                    
    #                 score_differences = torch.abs(original_scores - perturbed_scores)
    #                 max_changes = torch.max(score_differences, dim=1)[0]
                    
    #                 for (_, _, perturbed_part, original_label), max_change in zip(perturbed_batch, max_changes):
    #                     if max_change >= self.change_threshold:
    #                         ate_data.append((perturbed_part, original_label))

    #     self.ate_train_data = ate_data
    
    # def _prepare_ate_data(self, batch_size=32):
    #     train_loader = self.get_train_dataloader(batch_size=batch_size)
    #     vocab = self.get_vocab()
    #     ate_data = []

    #     self.model.eval()
    #     self.model.to(self.device)

    #     with torch.no_grad():
    #         for batch in tqdm(train_loader, desc="Preparing ATE data"):
    #             perturbed_batches = self._perturb_batch(batch, vocab)
                
    #             for batch_perturbed in perturbed_batches:
    #                 original_inputs = [item[0] for item in batch_perturbed]
    #                 perturbed_inputs = [item[1] for item in batch_perturbed]
                    
    #                 original_encodings = self.tokenizer(original_inputs, truncation=True, padding=True, return_tensors="pt")
    #                 perturbed_encodings = self.tokenizer(perturbed_inputs, truncation=True, padding=True, return_tensors="pt")
                    
    #                 original_scores = self.model(**{k: v.to(self.device) for k, v in original_encodings.items()})
    #                 perturbed_scores = self.model(**{k: v.to(self.device) for k, v in perturbed_encodings.items()})
                    
    #                 # If the scores are not already probabilities, apply softmax
    #                 if original_scores.dim() > 1:
    #                     original_scores = torch.nn.functional.softmax(original_scores, dim=-1)
    #                     perturbed_scores = torch.nn.functional.softmax(perturbed_scores, dim=-1)
                    
    #                 score_differences = torch.abs(original_scores - perturbed_scores)
    #                 max_changes = torch.max(score_differences, dim=1)[0]
                    
    #                 for (_, _, perturbed_part, original_label), max_change in zip(batch_perturbed, max_changes):
    #                     if max_change >= self.change_threshold:
    #                         ate_data.append((perturbed_part, original_label))

    #     self.ate_train_data = ate_data
    #     print(f"Generated {len(ate_data)} ATE training samples.")


    def _pre_compute_all_perturbations(self, batch_size=32):
        train_loader = self.get_train_dataloader(batch_size=batch_size)
        vocab = self.get_vocab()
        all_perturbations = []

        for batch in tqdm(train_loader, desc="Pre-computing perturbations"):
            perturbed_batches = self._perturb_batch(batch, vocab)
            all_perturbations.extend(perturbed_batches)

        return all_perturbations
    

    def _prepare_ate_data(self, batch_size=32):
        train_loader = self.get_train_dataloader(batch_size=batch_size)
        vocab = self.get_vocab()
        ate_data = []

        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Preparing ATE data"):
                # Generate all perturbations for the batch
                all_perturbed = self._perturb_batch(batch, vocab)
                
                # Flatten the list of perturbations
                flat_perturbed = [item for sublist in all_perturbed for item in sublist]
                
                # Separate original and perturbed inputs
                original_inputs = [item[0] for item in flat_perturbed]
                perturbed_inputs = [item[1] for item in flat_perturbed]
                
                # Tokenize in batches
                original_encodings = self.tokenizer(original_inputs, truncation=True, padding=True, return_tensors="pt")
                perturbed_encodings = self.tokenizer(perturbed_inputs, truncation=True, padding=True, return_tensors="pt")
                
                # Process in smaller batches to avoid OOM errors
                batch_size = 128
                for i in range(0, len(original_inputs), batch_size):
                    orig_batch = {k: v[i:i+batch_size].to(self.device) for k, v in original_encodings.items()}
                    pert_batch = {k: v[i:i+batch_size].to(self.device) for k, v in perturbed_encodings.items()}
                    
                    original_scores = self.model(**orig_batch)
                    perturbed_scores = self.model(**pert_batch)
                    
                    if hasattr(original_scores, 'logits'):
                        original_scores = original_scores.logits
                        perturbed_scores = perturbed_scores.logits
                    
                    score_differences = torch.abs(original_scores - perturbed_scores)
                    max_changes = torch.max(score_differences, dim=1)[0]
                    
                    for (_, _, perturbed_part, original_label), max_change in zip(flat_perturbed[i:i+batch_size], max_changes):
                        if max_change >= self.change_threshold:
                            ate_data.append((perturbed_part, original_label))

        self.ate_train_data = ate_data
        print(f"Generated {len(ate_data)} ATE training samples.")
    
    # def _prepare_ate_data(self, batch_size=32):
    #     print("Pre-computing perturbations...")
    #     all_perturbations = self._pre_compute_all_perturbations(batch_size=batch_size)
        
    #     print(f"Total perturbations generated: {sum(len(batch) for batch in all_perturbations)}")
        
    #     ate_data = []
    #     self.model.eval()
    #     self.model.to(self.device)

    #     with torch.no_grad():
    #         for perturbed_batch in tqdm(all_perturbations, desc="Scoring perturbations"):
    #             original_inputs = [item[0] for item in perturbed_batch]
    #             perturbed_inputs = [item[1] for item in perturbed_batch]
                
    #             original_encodings = self.tokenizer(original_inputs, truncation=True, padding=True, return_tensors="pt")
    #             perturbed_encodings = self.tokenizer(perturbed_inputs, truncation=True, padding=True, return_tensors="pt")
                
    #             original_scores = self.model(**{k: v.to(self.device) for k, v in original_encodings.items()})
    #             perturbed_scores = self.model(**{k: v.to(self.device) for k, v in perturbed_encodings.items()})
                
    #             if hasattr(original_scores, 'logits'):
    #                 original_scores = original_scores.logits
    #                 perturbed_scores = perturbed_scores.logits
                
    #             score_differences = torch.abs(original_scores - perturbed_scores)
    #             max_changes = torch.max(score_differences, dim=1)[0]
                
    #             for (_, _, perturbed_part, original_label), max_change in zip(perturbed_batch, max_changes):
    #                 if max_change >= self.change_threshold:
    #                     ate_data.append((perturbed_part, original_label))

    #     self.ate_train_data = ate_data
    #     print(f"Generated {len(ate_data)} ATE training samples.")
        
    
    def get_ate_train_dataloader(self, batch_size=32):
        if self.ate_train_data is None:
            self._prepare_ate_data()
        
        def collate_fn(batch):
            texts, labels = zip(*batch)
            encodings = self.tokenizer(list(texts), truncation=True, padding=True, return_tensors="pt")
            return {
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'labels': torch.tensor(labels, dtype=torch.long)
            }
        
        dataset = SimpleDataset(self.ate_train_data, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def get_dataloaders(self, tokenizer, batch_size):
        return self.get_ate_train_dataloader(batch_size), self.get_val_dataloader(batch_size)