import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Hyperparameters
NUM_POLICIES = 100  # Number of AI-generated policies to test
LEARNING_RATE = 0.001
BATCH_SIZE = 32

class PolicyDataset(Dataset):
    """Dataset to store and retrieve AI-generated policies and their simulated outcomes."""
    def __init__(self, num_policies=NUM_POLICIES):
        self.num_policies = num_policies
        self.policies = self.generate_policies()
    
    def generate_policies(self):
        """Generate random policy interventions for simulation."""
        policies = []
        for _ in range(self.num_policies):
            budget = np.random.uniform(100, 1000)  # In billions of yen
            duration = np.random.uniform(1, 10)  # Duration in years
            effect = np.random.uniform(0.5, 5.0)  # Expected birth rate increase per 1000 people
            policies.append((budget, duration, effect))
        return policies
    
    def __len__(self):
        return len(self.policies)

    def __getitem__(self, idx):
        budget, duration, effect = self.policies[idx]
        return torch.tensor([budget, duration]), torch.tensor([effect])

class PolicyImpactModel(nn.Module):
    """Simple neural network to model birth rate impact from policy interventions."""
    def __init__(self):
        super(PolicyImpactModel, self).__init__()
        self.fc1 = nn.Linear(2, 128)  # Adjust input size to match the number of input features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Trainer:
    """Handles training and evaluating the AI Scientist-generated policies."""
    def __init__(self, model, dataloader, learning_rate=LEARNING_RATE):
        self.model = model
        self.dataloader = dataloader
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def train(self, epochs=10):
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in self.dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs.float())
                loss = self.criterion(outputs, targets.float())
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(self.dataloader):.4f}")
    
    def evaluate(self, test_data):
        """Evaluates the model on test policies."""
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, _ in test_data:
                predictions.append(self.model(inputs.float()).item())
        return predictions

def preprocess_results(results):
    """Convert list of results to a dictionary format expected by the plotting function."""
    processed_results = {}
    for policy_result in results:
        policy_key = str(tuple(policy_result["policy"]))  # Convert tuple to string
        processed_results[policy_key] = {"means": policy_result["predicted_impact"]}
    return processed_results

def main():
    parser = argparse.ArgumentParser(description="Run birth rate policy experiment")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    dataset = PolicyDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = PolicyImpactModel()
    trainer = Trainer(model, dataloader)
    
    print("Training model...")
    trainer.train(epochs=30)
    
    print("Evaluating policies...")
    test_data = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions = trainer.evaluate(test_data)
    
    results = [{"policy": dataset.policies[i], "predicted_impact": predictions[i]} for i in range(len(predictions))]
    
    processed_results = preprocess_results(results)
    
    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(processed_results, f, indent=4)
    
    print("Experiment complete. Results saved.")

if __name__ == "__main__":
    main()
