import torch
from tqdm import tqdm

def test_nn(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, targets, _ in tqdm(test_loader, desc="Testing"):  # Ignore lengths
            inputs, targets = inputs.to(device).long(), targets.to(device).view(-1, 1)  # Ensure targets have shape (batch_size, 1)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            pred = (outputs > 0.5).float()
            correct += (pred == targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    return accuracy
