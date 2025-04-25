import torch
import deepchem as dc
import torch.nn as nn
import json
import os
from mtl_dnn_training_draft import DNNMultiTaskClassifier

# Load the model checkpoint first
checkpoint_path = r"C:\Users\Ford\Desktop\fyp\mtl_dnn\final_model.pth"
checkpoint = torch.load(checkpoint_path)

# Load the model architecture from the saved checkpoint
model_dir = os.path.dirname(checkpoint_path)
architecture_path = os.path.join(model_dir, "global_best_architecture.json")

with open(architecture_path, "r") as f:
    architecture = json.load(f)

# Extract exact architecture from saved config
n_features = architecture['n_features']
layer_sizes = architecture['layer_sizes']
dropouts = architecture['dropouts']
hidden_layers = architecture['hidden_layers']
hidden_layer_dropouts = architecture['hidden_layer_dropouts']

# Create model with exact same architecture as saved
model = DNNMultiTaskClassifier(
    n_features=n_features,
    layer_sizes=layer_sizes,
    dropouts=dropouts,
    hidden_layer_sizes=hidden_layers,
    hidden_layer_dropouts=hidden_layer_dropouts,
    n_tasks=12
)

# Load the state dict
model.load_state_dict(checkpoint)
model.eval()

# Define a function for making predictions
def predict(smiles):
    featurizer = dc.feat.CircularFingerprint()
    X = featurizer.featurize([smiles])
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X)
        # Apply sigmoid to each task's output
        predictions = [torch.sigmoid(output).item() for output in outputs]
        return predictions

# Define a function to interpret predictions
def interpret_predictions(predictions, threshold=0.3):  # Lower threshold
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    
    def get_interpretation(pred):
        if pred >= 0.7:
            return 'HIGH TOXICITY RISK', 'red'
        elif pred >= 0.5:
            return 'Moderate Toxic Risk', 'yellow'
        elif pred >= 0.3:
            return 'Low Toxic Risk', 'blue'
        return 'Likely Non-toxic', 'green'
    
    return {task: get_interpretation(pred) for task, pred in zip(tasks, predictions)}

def get_user_input():
    """Get SMILES string from user"""
    while True:
        smiles = input("\nEnter SMILES string (or 'q' to quit, Enter for example 'O'): ").strip()
        if smiles.lower() == 'q':
            return None
        if smiles or smiles == "O":
            return smiles
        print("Please enter a valid SMILES string.")

if __name__ == "__main__":
    print("\nMulti-Task Toxicity Prediction System")
    print("=" * 50)
    print("\nInterpretation Guide:")
    print("- Score >= 0.7: HIGH TOXICITY RISK")
    print("- Score >= 0.5: Moderate Toxic Risk")
    print("- Score >= 0.3: Low Toxic Risk")
    print("- Score < 0.3: Likely Non-toxic")
    
    while True:
        # Get user input
        smiles = get_user_input()
        if smiles is None:  # User chose to quit
            print("\nExiting program. Goodbye!")
            break
            
        print(f"\nMaking predictions for:")
        print(f"SMILES: {smiles}")
        print("-" * 50)
        
        try:
            # Make predictions
            predictions = predict(smiles)
            interpretation = interpret_predictions(predictions)
            
            # Show results in a formatted table
            print("\nPrediction Results:")
            print("-" * 75)
            print(f"{'Task':15} | {'Score':10} | {'Risk Assessment':30}")
            print("-" * 75)
            
            any_high_risk = False
            for task, (result, color) in interpretation.items():
                score = predictions[list(interpretation.keys()).index(task)]
                print(f"{task:15} | {score:10.4f} | {result:30}")
                if score >= 0.7:
                    any_high_risk = True
            print("-" * 75)
            
            if any_high_risk:
                print("\n⚠️ WARNING: High toxicity risk detected in one or more categories!")
                print("This compound should be treated with caution.")
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
        
        # Ask if user wants to continue
        choice = input("\nPress Enter to make another prediction, or 'q' to quit: ").strip().lower()
        if choice == 'q':
            print("\nExiting program. Goodbye!")
            break
