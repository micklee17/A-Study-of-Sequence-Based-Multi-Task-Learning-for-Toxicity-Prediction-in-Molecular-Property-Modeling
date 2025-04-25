import torch
import deepchem as dc
import torch.nn as nn
import json
import os
from stl_dnn_train_draft import DNNSingleTaskClassifier

TASKS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
         'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']

def display_available_tasks():
    """Display all available tasks with numbers"""
    print("\nAvailable Tasks:")
    print("-" * 50)
    for i, task in enumerate(TASKS, 1):
        print(f"{i}. {task}")
    print("-" * 50)

def get_user_input():
    """Get SMILES string and task selection from user"""
    print("\nToxicity Prediction System")
    print("=" * 50)
    
    # Get SMILES input
    while True:
        smiles = input("\nEnter SMILES string (or 'q' to quit, Enter for example 'O'): ").strip()
        if smiles.lower() == 'q':
            return None, None
        if smiles or smiles == "O":
            break
        print("Please enter a valid SMILES string.")
    
    # Display tasks and get selection
    while True:
        display_available_tasks()
        try:
            task_num = input("\nSelect task number (1-12, or 'q' to quit): ").strip()
            if task_num.lower() == 'q':
                return None, None
            task_num = int(task_num)
            if 1 <= task_num <= len(TASKS):
                return smiles, TASKS[task_num - 1]
            print(f"Please enter a number between 1 and {len(TASKS)}")
        except ValueError:
            print("Please enter a valid number")

def load_model_for_task(task):
    """Load model and architecture for a specific task"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, f"best_dnn_single_task_model_{task}.pth")
    
    # Load checkpoint
    checkpoint = torch.load(model_path)
    architecture = checkpoint['architecture']
    
    # Create model with saved architecture
    model = DNNSingleTaskClassifier(
        n_features=architecture['n_features'],
        layer_sizes=architecture['layer_sizes'],
        dropouts=architecture['dropouts'],
        hidden_layers=architecture['hidden_layers'],
        hidden_layer_dropouts=architecture['hidden_layer_dropouts']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict(smiles, task):
    """Make prediction for a specific task"""
    # Load task-specific model
    model = load_model_for_task(task)
    
    # Featurize input
    featurizer = dc.feat.CircularFingerprint()
    X = featurizer.featurize([smiles])
    X = torch.tensor(X, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        output = model(X)
        prediction = torch.sigmoid(output).item()
    return prediction

def interpret_prediction(prediction, threshold=0.5):
    """Interpret the prediction result"""
    return 'Toxic' if prediction >= threshold else 'Non-toxic'

def predict_all_tasks(smiles):
    """Make predictions for all tasks"""
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    
    results = {}
    for task in tasks:
        try:
            prediction = predict(smiles, task)
            interpretation = interpret_prediction(prediction)
            results[task] = {
                'prediction': prediction,
                'interpretation': interpretation
            }
        except Exception as e:
            print(f"Error predicting for task {task}: {str(e)}")
            results[task] = {'error': str(e)}
    
    return results

if __name__ == "__main__":
    print("\nSingle-Task Toxicity Prediction System")
    print("=" * 50)
    
    while True:
        # Get user input
        smiles, task = get_user_input()
        if smiles is None:  # User chose to quit
            print("\nExiting program. Goodbye!")
            break
            
        print(f"\nPredicting toxicity for:")
        print(f"SMILES: {smiles}")
        print(f"Task: {task}")
        print("-" * 50)
        
        try:
            # Make prediction
            prediction = predict(smiles, task)
            interpretation = interpret_prediction(prediction)
            
            # Show results
            print("\nResults:")
            print(f"Prediction Score: {prediction:.4f}")
            print(f"Interpretation: {interpretation}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
        
        # Ask if user wants to continue
        if input("\nMake another prediction? (y/n): ").lower() != 'y':
            print("\nExiting program. Goodbye!")
            break
