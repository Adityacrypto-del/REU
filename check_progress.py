import os
import json
import glob

checkpoints = glob.glob('src/results/checkpoints/history_epoch_*.json')
if not checkpoints:
    print("No training checkpoints found yet.")
    exit(0)

# Sort checkpoints by epoch number
checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
latest_file = checkpoints[-1]
epoch = int(latest_file.split('_')[-1].split('.')[0])

with open(latest_file) as f:
    h = json.load(f)
    
val_accs = h['val_accs'][-1]
train_loss = h['train_loss'][-1]
val_loss = h['val_loss'][-1]

print("="*60)
print(f"Neural Collapse Training Progress: Epoch {epoch} / 100")
print("="*60)
print(f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
print("\nValidation Accuracies per Exit:")
for i, acc in enumerate(val_accs):
    print(f"  Exit {i+1} (Layer {i+1}): {acc:.2%}")
print("="*60)
