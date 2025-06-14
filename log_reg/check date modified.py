# check_models.py
import os
from datetime import datetime
import torch

def check_model_timestamps():
    """Check timestamps and details of all model files"""
    print("MODEL FILE ANALYSIS:")
    print("="*60)
    
    model_dir = 'models'
    if not os.path.exists(model_dir):
        print("Models directory not found!")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if not model_files:
        print("No model files found!")
        return
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)), reverse=True)
    
    print(f"{'Filename':<35} | {'Modified':<19} | {'Size (MB)':<8} | {'Type'}")
    print("-" * 75)
    
    for file in model_files:
        file_path = os.path.join(model_dir, file)
        timestamp = os.path.getmtime(file_path)
        readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        file_size = os.path.getsize(file_path) / (1024*1024)
        
        # Determine file type
        if 'best' in file:
            file_type = "Best"
        elif 'final' in file:
            file_type = "Final"
        else:
            file_type = "Other"
        
        print(f"{file:<35} | {readable_time} | {file_size:>7.2f} | {file_type}")

def check_model_details():
    """Check training details stored in model files"""
    print("\nMODEL TRAINING DETAILS:")
    print("="*60)
    
    for model_name in ['root', 'stem', 'branch']:
        print(f"\n{model_name.upper()} FAILURE MODEL:")
        print("-" * 30)
        
        # Check final model
        final_path = f'models/tree_{model_name}_failure_final.pth'
        try:
            checkpoint = torch.load(final_path, map_location='cpu')
            
            if 'epochs_trained' in checkpoint:
                print(f"  Epochs trained: {checkpoint['epochs_trained']}")
            if 'early_stopped' in checkpoint:
                print(f"  Early stopped: {checkpoint['early_stopped']}")
            if 'best_val_loss' in checkpoint:
                print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
            if 'history' in checkpoint:
                history = checkpoint['history']
                if 'val_acc' in history and history['val_acc']:
                    best_acc = max(history['val_acc'])
                    final_acc = history['val_acc'][-1]
                    print(f"  Best val acc: {best_acc:.2f}%")
                    print(f"  Final val acc: {final_acc:.2f}%")
                    
        except FileNotFoundError:
            print(f"  File not found: {final_path}")
        except Exception as e:
            print(f"  Error loading: {e}")

if __name__ == "__main__":
    check_model_timestamps()
    check_model_details()