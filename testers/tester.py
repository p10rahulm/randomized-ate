import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Tuple, Union, Any

class Tester:
    def __init__(self, model: nn.Module, data_module: Union[Any, None], batch_size: int = 32,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.data_module = data_module
        self.batch_size = batch_size
        self.device = device
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    def test(self) -> Tuple[float, float, float, float, float]:
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        # Get the test dataloader
        test_loader = self.data_module.get_test_dataloader(self.model.tokenizer, self.batch_size)

        progress_bar = tqdm(test_loader, desc="Testing")
        with torch.no_grad():
            for batch in progress_bar:
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                else:
                    input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids, attention_mask)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        return total_loss / len(test_loader), accuracy, precision, recall, f1

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        print(f"Model loaded from {path}")