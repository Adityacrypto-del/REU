import os
import json
import glob

def check_dataset_progress(name, folder):
    checkpoints = glob.glob(os.path.join(folder, 'history_epoch_*.json'))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    latest_file = checkpoints[-1]
    epoch = int(latest_file.split('_')[-1].split('.')[0])
    with open(latest_file) as f:
        h = json.load(f)
    return {
        'name': name,
        'epoch': epoch,
        'train_loss': h['train_loss'][-1],
        'val_loss': h['val_loss'][-1],
        'val_accs': h['val_accs'][-1],
    }

datasets = [
    ('Flowers-102 (Complex, K=102)', 'src/results/checkpoints'),
    ('CIFAR-10 (Simple, K=10)', 'src/results/cifar10/checkpoints'),
]

for name, folder in datasets:
    res = check_dataset_progress(name, folder)
    if res:
        print("=" * 60)
        print(f"[{res['name']}] Progress: Epoch {res['epoch']} / 100")
        print("=" * 60)
        print(f"Train Loss: {res['train_loss']:.4f}  |  Val Loss: {res['val_loss']:.4f}")
        print("\nValidation Accuracies per Exit:")
        for i, acc in enumerate(res['val_accs']):
            print(f"  Exit {i+1} (Layer {i+1}): {acc:.2%}")
        print("=" * 60 + "\n")

