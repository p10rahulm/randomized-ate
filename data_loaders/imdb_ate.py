import torch
from torch.utils.data import DataLoader
from data_loaders.imdb import IMDBDataModule
import random
import numpy as np
from tqdm import tqdm
from data_loaders.simple_dataloader import SimpleDataset
import spacy
import string
from utilities.word_utils import load_spacy_model
import os
import pickle
from utilities.word_utils import extract_phrases, remove_punctuation_phrases
from inspect import signature  # Import the inspect module
from typing import Dict, Any



class ATEDataModule(IMDBDataModule):
    def __init__(self, classification_word: str, model: Any, model_config: Dict[str, Any], 
                 perturbation_rate: float = 0.5, num_perturbations: int = 25, 
                 n_gram_length: int = 5, change_threshold: float = 0.5):
        super().__init__(classification_word)
        self.model = model
        self.perturbation_rate = perturbation_rate
        self.num_perturbations = num_perturbations
        self.n_gram_length = n_gram_length
        self.change_threshold = change_threshold
        self.ate_train_data = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = model.tokenizer
        self.nlp = load_spacy_model("en_core_web_sm")
        self.vocab = list(self.tokenizer.get_vocab().keys())
        self.model_config = model_config  # Should contain 'model_name', 'num_epochs', and 'learning_rate'
        

    
    def perturb_text(self, text):
        phrases = remove_punctuation_phrases(extract_phrases(text))
        total_phrases = len(phrases)
        
        # Group phrases into sets of n_gram_length
        grouped_phrases = [' '.join(phrases[i:i+self.n_gram_length]) for i in range(0, total_phrases, self.n_gram_length)]
        
        # Create multiple copies
        phrases_n_copies = [grouped_phrases.copy() for _ in range(self.num_perturbations)]
        
        # Calculate number of phrases to perturb
        num_to_perturb = min(int(len(grouped_phrases)), max(3, int(len(grouped_phrases) * self.perturbation_rate)))
        
        # Generate noise tokens
        noise_tokens = np.random.choice(self.vocab, size=(self.num_perturbations, num_to_perturb))
        
        # Create perturbation masks
        masks = np.zeros((self.num_perturbations, len(grouped_phrases)), dtype=bool)
        for mask in masks:
            mask[np.random.choice(len(grouped_phrases), num_to_perturb, replace=False)] = True
        
        perturbed_versions = []
        for phrases_copy, noise, mask in zip(phrases_n_copies, noise_tokens, masks):
            perturbed_part = ' '.join(np.array(phrases_copy)[mask])
            phrases_copy = np.array(phrases_copy)
            phrases_copy[mask] = noise
            perturbed_text = ' '.join(phrases_copy)
            perturbed_versions.append((perturbed_text, perturbed_part))
        
        return perturbed_versions

    def prepare_data(self, batch_size=32):
        super().load_data()  # Load the IMDB data
        self._prepare_ate_data(batch_size=batch_size)

    def _generate_perturbations(self):
        print("Generating perturbations...")
        train_data = self.preprocess()
        
        all_perturbations = []
        for text, label in tqdm(train_data, desc="Generating perturbations"):
            perturbed_versions = self.perturb_text(text)
            all_perturbations.extend([(text, perturbed, perturbed_part, label) 
                                      for perturbed, perturbed_part in perturbed_versions])
        
        print(f"Total perturbations generated: {len(all_perturbations)}")
        return all_perturbations

    def _score_perturbations(self, all_perturbations, batch_size=32):
        print("Scoring perturbations...")
        ate_data = []
        self.model.eval()
        self.model.to(self.device)
        
        # Retrieve the argument names of the model's forward method using the inspect module
        model_forward_signature = signature(self.model.forward)
        model_forward_arg_names = model_forward_signature.parameters.keys()

        with torch.no_grad():
            for i in tqdm(range(0, len(all_perturbations), batch_size), desc="Scoring perturbations"):
                batch = all_perturbations[i:i+batch_size]
                original_inputs = [item[0] for item in batch]
                perturbed_inputs = [item[1] for item in batch]
                
                original_encodings = self.tokenizer(original_inputs, truncation=True, padding=True, return_tensors="pt")
                perturbed_encodings = self.tokenizer(perturbed_inputs, truncation=True, padding=True, return_tensors="pt")

                # Filter tokenizer outputs to include only the arguments accepted by the model's forward method
                # This prevents passing unexpected arguments like 'token_type_ids' to models that don't use them
                inputs = {k: v.to(self.device) for k, v in original_encodings.items() if k in model_forward_arg_names}
                original_scores = self.model(**inputs)

                inputs = {k: v.to(self.device) for k, v in perturbed_encodings.items() if k in model_forward_arg_names}
                perturbed_scores = self.model(**inputs)
                              
                # original_scores = self.model(**{k: v.to(self.device) for k, v in original_encodings.items()})
                # perturbed_scores = self.model(**{k: v.to(self.device) for k, v in perturbed_encodings.items()})
                
                if hasattr(original_scores, 'logits'):
                    original_scores = original_scores.logits
                    perturbed_scores = perturbed_scores.logits
                
                score_differences = torch.abs(original_scores - perturbed_scores)
                max_changes = torch.max(score_differences, dim=1)[0]
                
                for (_, _, perturbed_part, label), max_change in zip(batch, max_changes):
                    if max_change >= self.change_threshold:
                        ate_data.append((perturbed_part, label))

        print(f"Generated {len(ate_data)} ATE training samples.")
        return ate_data

    def _prepare_ate_data(self, batch_size: int = 32) -> None:
        model_name = self.model_config['model_name']
        num_epochs = self.model_config['num_epochs']
        learning_rate = self.model_config['learning_rate']

        perturbations_dir = f'saved/imdb_perturbations_{self.num_perturbations}_{self.perturbation_rate}'
        perturbations_file = os.path.join(perturbations_dir, 'perturbations.pkl')
        
        ate_dir = f'saved/imdb_ate_{self.num_perturbations}_{self.perturbation_rate}'
        os.makedirs(ate_dir, exist_ok=True)
        ate_file = os.path.join(ate_dir, f'{model_name}_base_e{num_epochs}_lr{learning_rate}.pkl')

        if os.path.exists(ate_file):
            print(f"Loading pre-computed ATE data from {ate_file}")
            with open(ate_file, 'rb') as f:
                self.ate_train_data = pickle.load(f)
        else:
            if os.path.exists(perturbations_file):
                print(f"Loading pre-computed perturbations from {perturbations_file}")
                with open(perturbations_file, 'rb') as f:
                    all_perturbations = pickle.load(f)
            else:
                all_perturbations = self._generate_perturbations()
                os.makedirs(perturbations_dir, exist_ok=True)
                with open(perturbations_file, 'wb') as f:
                    pickle.dump(all_perturbations, f)

            self.ate_train_data = self._score_perturbations(all_perturbations, batch_size)
            
            with open(ate_file, 'wb') as f:
                pickle.dump(self.ate_train_data, f)

    def get_ate_train_dataloader(self, batch_size=32):
        if self.ate_train_data is None:
            raise ValueError("ATE data has not been prepared. Call prepare_data() first.")
        
        dataset = SimpleDataset(self.ate_train_data, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.ate_collate_fn)

    def get_dataloaders(self, batch_size):
        return self.get_ate_train_dataloader(batch_size), self.get_val_dataloader(batch_size)
    
    def ate_collate_fn(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_masks = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        return input_ids, attention_masks, labels
   
    def get_full_train_dataloader(self, batch_size=32):
        if self.ate_train_data is None:
            raise ValueError("ATE data has not been prepared. Call prepare_data() first.")
        
        # Combine train and validation data
        full_train_data = self.ate_train_data + [
            (text, label) for text, label in self.preprocess()[len(self.train_dataset):]
        ]
        
        dataset = SimpleDataset(full_train_data, self.tokenizer)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=self.ate_collate_fn)