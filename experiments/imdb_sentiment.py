import os
from pathlib import Path
import sys
import csv
from datetime import datetime
import argparse
import logging

# Add project root to system path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.model_variations import model_variations
from data_loaders.imdb import IMDBDataModule
from data_loaders.imdb_ate import ATEDataModule
from trainers.trainer import Trainer
from testers.tester import Tester
from optimizers.optimizer_params import optimizer_configs
from utilities.model_utilities import save_trained_model, load_trained_model
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_experiment(model: Any, trainer: Trainer, tester: Tester, num_epochs: int, writer: csv.DictWriter, csvfile,
                   model_type: str, model_name: str, learning_rate: float, run_training: bool = True, 
                   run_testing: bool = True) -> None:
    training_history = [-1]
    model_path = ""
    if run_training:
        training_history = trainer.train_on_full_dataset(num_epochs=num_epochs)
        model_dir = f"{model_name}_{model_type}_e{num_epochs}_lr{learning_rate}"
        model_path = save_trained_model(model, 'sentiment_imdb', model_dir, f"{model_type}")

    if run_testing:
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = tester.test()

        writer.writerow({
            'model': model_name,
            'type': model_type,
            'epochs': num_epochs,
            'final_train_loss': training_history[-1],
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'model_path': model_path
        })
        csvfile.flush()


def run_imdb_sentiment_experiment(args):
    # Experiment parameters
    models = args.models
    epochs = args.epochs
    batch_size = args.batch_size
    patience = args.patience
    classification_word = args.classification_word or "Sentiment"
    preload_model = args.preload_model or False
    print(f"preload_model={preload_model}")
    

    # Hyperparameters
    optimizer_name = args.optimizer
    hidden_layer = args.hidden_layer
    learning_rate = args.learning_rate

    # Prepare data loader
    data_module = IMDBDataModule(classification_word)

    # Prepare results file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "outputs/imdb_sentiment_classifier_with_ate"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"results_{timestamp}.csv")

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'type', 'epochs', 'final_train_loss', 'test_loss', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'model_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_name in models:
            for num_epochs in epochs:
                logging.info(f"Running experiment: {model_name}, epochs={num_epochs}")

                
                # Create base model
                model_preloaded_flag = False
                if preload_model:
                    try:
                        base_model_dir = f"{model_name}_base_e{num_epochs}_lr{learning_rate}"
                        base_model = load_trained_model(model_variations[model_name][hidden_layer], 'sentiment_imdb', base_model_dir,
                                                        classification_word=classification_word, freeze_encoder=True)
                        if base_model is not None:
                            logging.info(f"Loaded pre-trained model for {model_name}. Skipping training.")
                            model_preloaded_flag = True
                        else:
                            logging.info(f"No pre-trained model found for {model_name}. Training from scratch.")                            
                    except Exception as e:
                        logging.error(f"Error loading pre-trained model for {model_name}: {str(e)}. Training from scratch.")
                        

                if not model_preloaded_flag:
                    base_model = model_variations[model_name][hidden_layer](classification_word, freeze_encoder=True)

                # Update optimizer config
                optimizer_config = optimizer_configs[optimizer_name].copy()
                optimizer_config['params'] = optimizer_config['params'].copy()
                optimizer_config['params']['lr'] = learning_rate

                # Create trainer for base model
                base_trainer = Trainer(base_model, data_module, optimizer_name=optimizer_name,
                                  optimizer_params=optimizer_config['params'],
                                  batch_size=batch_size, num_epochs=num_epochs, patience=patience)
                

                base_tester = Tester(base_model, data_module, batch_size=batch_size)

                # Run experiment for base model
                run_experiment(base_model, base_trainer, base_tester, num_epochs, writer, csvfile, 'base', 
                               model_name, learning_rate, run_training=not model_preloaded_flag, run_testing=True)
                
                # PART 2: ATE BASED MODEL

                # Create ATE model
                ate_model = model_variations[model_name][hidden_layer](classification_word, freeze_encoder=True)
                # Create New ATE Data
                model_config = {
                    'model_name': model_name,
                    'num_epochs': num_epochs,
                    'learning_rate': learning_rate
                }
                
                # Create New ATE Data
                ate_data_module = ATEDataModule(
                    classification_word=args.classification_word,
                    model=base_model,
                    model_config=model_config,
                    perturbation_rate=args.perturbation_rate,
                    num_perturbations=args.num_perturbations,
                    n_gram_length=args.n_gram_length,
                    change_threshold=args.change_threshold
                )
                
                # Prepare perturbed data
                ate_data_module.prepare_data(batch_size=args.ate_batch_size)

                # Create trainer for ATE model
                ate_trainer = Trainer(ate_model, ate_data_module, optimizer_name=optimizer_name,
                                      optimizer_params=optimizer_config['params'],
                                      batch_size=batch_size, num_epochs=num_epochs, patience=patience)

                # Test the ATE model
                ate_tester = Tester(ate_model, data_module, batch_size=batch_size)
                
                run_experiment(ate_model, ate_trainer, ate_tester, num_epochs, writer, csvfile, 
                               'ate', model_name, learning_rate, run_training=True, run_testing=True)
                

        print(f"Experiments completed. Results saved to {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IMDB Sentiment Experiment")
    parser.add_argument("--models", nargs="+", default=["roberta", "albert", "distilbert", "bert", "deberta_small", "t5"],
                        help="List of models to run experiments on")
    parser.add_argument("--epochs", nargs="+", type=int, default=[1,2,5],
                        help="List of epoch numbers to run experiments with")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--optimizer", default="adamw", help="Optimizer to use")
    parser.add_argument("--hidden_layer", default="1_hidden", help="Hidden layer configuration")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--perturbation_rate", type=float, default=0.5, help="Perturbation rate for ATE")
    parser.add_argument("--num_perturbations", type=int, default=25, help="Number of perturbations for ATE")
    parser.add_argument("--n_gram_length", type=int, default=5, help="N-gram length for ATE")
    parser.add_argument("--change_threshold", type=float, default=0.5, help="Change threshold for ATE")
    parser.add_argument("--ate_batch_size", type=int, default=256, help="Batch size for ATE data preparation")
    parser.add_argument("--preload_model", action="store_true", help="Load the latest trained model instead of training from scratch")
    parser.add_argument("--classification_word", default="Sentiment", help="Classification word")


    args = parser.parse_args()
    run_imdb_sentiment_experiment(args)