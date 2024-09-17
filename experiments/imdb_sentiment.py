import os
from pathlib import Path
import sys
import csv
from datetime import datetime

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

from models.model_variations import model_variations
from data_loaders.imdb import IMDBDataModule
from data_loaders.imdb_ate import ATEDataModule
from trainers.trainer import Trainer
from testers.tester import Tester
from optimizers.optimizer_params import optimizer_configs


def run_imdb_sentiment_experiment():
    # Experiment parameters
    models = ["roberta", "bert", "albert", "distilbert", "electra_small_discriminator"]
    classification_word = "Sentiment"
    epochs = [5, 10, 20]
    batch_size = 1024
    patience = 3

    # Hyperparameters
    optimizer_name = "adamw"
    hidden_layer = "1_hidden"
    learning_rate = 0.0001

    # Prepare data loader
    data_module = IMDBDataModule(classification_word)

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/imdb_sentiment_classifier_with_ate"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.csv")

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'type', 'epochs', 'final_train_loss', 'test_loss', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in models:
            for num_epochs in epochs:
                print(f"Running experiment: {model_name}, epochs={num_epochs}")

                # Create base model
                base_model = model_variations[model_name][hidden_layer](classification_word, freeze_encoder=True)

                # Update optimizer config
                optimizer_config = optimizer_configs[optimizer_name].copy()
                optimizer_config['params'] = optimizer_config['params'].copy()
                optimizer_config['params']['lr'] = learning_rate

                # Create trainer for base model
                trainer = Trainer(base_model, data_module, optimizer_name=optimizer_name,
                                  optimizer_params=optimizer_config['params'],
                                  batch_size=batch_size, num_epochs=num_epochs, patience=patience)

                # Train on full dataset
                training_history = trainer.train_on_full_dataset(num_epochs=num_epochs)

                # Test the base model
                tester = Tester(base_model, data_module, batch_size=batch_size)
                test_loss, test_accuracy, test_precision, test_recall, test_f1 = tester.test()

                # Write base model results
                writer.writerow({
                    'model': model_name,
                    'type': 'base',
                    'epochs': num_epochs,
                    'final_train_loss': training_history[-1],
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1
                })
                csvfile.flush()

                # PART 2: ATE BASED MODEL

                # Create ATE model
                ate_model = model_variations[model_name][hidden_layer](classification_word, freeze_encoder=True)

                # Create New ATE Data
                ate_data_module = ATEDataModule(
                    classification_word=classification_word,
                    model=base_model,
                    perturbation_rate=0.5,
                    num_perturbations=25,
                    n_gram_length=5,
                    change_threshold=0.5
                )
                
                # Prepare perturbed data
                ate_data_module.prepare_data()

                # Create trainer for ATE model
                ate_trainer = Trainer(ate_model, ate_data_module, optimizer_name=optimizer_name,
                                      optimizer_params=optimizer_config['params'],
                                      batch_size=batch_size, num_epochs=num_epochs, patience=patience)

                # Train ATE model
                ate_training_history = ate_trainer.train_on_full_dataset(num_epochs=num_epochs)

                # Test the ATE model
                ate_tester = Tester(ate_model, data_module, batch_size=batch_size)
                ate_test_loss, ate_test_accuracy, ate_test_precision, ate_test_recall, ate_test_f1 = ate_tester.test()

                # Write ATE model results
                writer.writerow({
                    'model': model_name,
                    'type': 'ate',
                    'epochs': num_epochs,
                    'final_train_loss': ate_training_history[-1],
                    'test_loss': ate_test_loss,
                    'test_accuracy': ate_test_accuracy,
                    'test_precision': ate_test_precision,
                    'test_recall': ate_test_recall,
                    'test_f1': ate_test_f1
                })
                csvfile.flush()

        print(f"Experiments completed. Results saved to {results_file}")

if __name__ == "__main__":
    run_imdb_sentiment_experiment()
    