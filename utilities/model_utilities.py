import os
from datetime import datetime
import torch
from utilities.file_utilities import get_latest_file


def save_trained_model(model, dataset_name, model_name, additional_info=None):
    """
    Save a trained model to a specified path.

    Args:
    model (torch.nn.Module): The trained model to save.
    dataset_name (str): Name of the dataset (e.g., 'sentiment_imdb').
    model_name (str): Name of the model architecture (e.g., 'roberta').
    additional_info (str, optional): Additional information to include in the filename.

    Returns:
    str: The path where the model was saved.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the directory if it doesn't exist
    save_dir = os.path.join('trained_models', dataset_name, model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Create the filename
    filename = f"{timestamp}.pth"
    if additional_info:
        filename = f"{timestamp}_{additional_info}.pth"
    
    # Full path
    full_path = os.path.join(save_dir, filename)
    
    # Save the model
    torch.save(model.state_dict(), full_path)
    
    print(f"Model saved to {full_path}")
    return full_path


def load_trained_model(model_class, dataset_name, model_name, classification_word='Sentiment', freeze_encoder=True):
    """
    Load the latest trained model from the specified dataset and model name.

    Args:
    model_class (type): The class of the model to be loaded.
    dataset_name (str): Name of the dataset (e.g., 'sentiment_imdb').
    model_name (str): Name of the model architecture (e.g., 'roberta').

    Returns:
    torch.nn.Module: The loaded model, or None if no model file is found.
    """
    model_dir = os.path.join('trained_models', dataset_name, model_name)
    latest_model_path = get_latest_file(model_dir)
    
    if latest_model_path is None:
        print(f"No trained model found for {dataset_name} - {model_name}")
        return None

    model = model_class(cw=classification_word, freeze_encoder=freeze_encoder)  # Initialize the model
    model.load_state_dict(torch.load(latest_model_path))
    model.eval()  # Set the model to evaluation mode
    
    print(f"Loaded model from {latest_model_path}")
    return model

