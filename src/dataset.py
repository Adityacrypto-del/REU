import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_dataloaders(dataset_name='flowers102', batch_size=32, root_dir='./data'):
    """
    Downloads and prepares the specified dataset with data augmentation.
    Supported: 'flowers102' (complex fine-grained) and 'cifar10' (simple baseline).
    Returns: train_loader, val_loader, class_priors (for logit adjustment)
    """
    # Standard ResNet ImageNet normalizations and augmentations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    if dataset_name.lower() == 'flowers102':
        # Load Oxford Flowers 102
        train_dataset = datasets.Flowers102(root=root_dir, split="train", download=True, transform=train_transform)
        val_dataset = datasets.Flowers102(root=root_dir, split="test", download=True, transform=val_transform)
        num_classes = 102
    elif dataset_name.lower() == 'cifar10':
        # Load CIFAR-10 as a simple geometric complexity baseline (Munn et al. 2024)
        train_dataset = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=train_transform)
        val_dataset = datasets.CIFAR10(root=root_dir, train=False, download=True, transform=val_transform)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Calculate class priors for Multiplicative Logit Adjustment (Hasegawa & Sato 2024)
    # We count the frequency of each class in the training set
    import torch
    targets = []
    if hasattr(train_dataset, '_labels'):
        targets = train_dataset._labels
    elif hasattr(train_dataset, 'targets'):
        targets = train_dataset.targets
    
    if len(targets) > 0:
        class_counts = torch.bincount(torch.tensor(targets), minlength=num_classes).float()
        class_priors = class_counts / class_counts.sum()
    else:
        # Fallback to uniform priors if targets aren't easily accessible
        class_priors = torch.ones(num_classes) / num_classes

    # pin_memory only benefits CUDA; MPS doesn't support it
    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_pin)

    return train_loader, val_loader, class_priors

if __name__ == "__main__":
    # Test script to verify things work locally
    print("Testing data loader...")
    train_loader, val_loader, class_priors = get_dataloaders(batch_size=16)
    print(f"Training batches: {len(train_loader)} (approx {len(train_loader)*16} images)")
    print(f"Validation batches: {len(val_loader)} (approx {len(val_loader)*16} images)")
    print("Data loading success!")
