import torch
import torch.nn as nn
import torchvision.models as models

class EarlyExitResNet(nn.Module):
    def __init__(self, num_classes=102, pretrained=True):
        super(EarlyExitResNet, self).__init__()
        
        # Load the base model. Default to True to leverage ImageNet features initially
        # Using the modern weights= API (pretrained= was deprecated in torchvision 0.13)
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Remove the final fully connected layer from the backbone
        self.backbone.fc = nn.Identity()

        # Define early exits classifiers for intermediate layers
        # Layer 1 outputs 64 channels
        self.exit1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        
        # Layer 2 outputs 128 channels
        self.exit2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
        
        # Layer 3 outputs 256 channels
        self.exit3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        
        # Layer 4 (Final Exit) outputs 512 channels
        self.exit4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # We process the image layer by layer and run the classifier at each block boundary
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        out1 = self.exit1(x1)

        x2 = self.backbone.layer2(x1)
        out2 = self.exit2(x2)

        x3 = self.backbone.layer3(x2)
        out3 = self.exit3(x3)

        x4 = self.backbone.layer4(x3)
        out4 = self.exit4(x4)
        
        # Return tuples of (Predictions) and (Features)
        # We need the features to perform the Neural Collapse mathematics later
        logits = (out1, out2, out3, out4)
        features = (x1, x2, x3, x4)
        
        return logits, features

    def forward_early_exit(self, x, confidence_threshold=0.9):
        """
        Actual Early Exiting for inference time.
        Saves compute by completely halting the forward pass if a shallow layer is highly confident.
        Returns the prediction and the layer number it exited at.
        """
        # Initial stem block
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # --- LAYER 1 ---
        x1 = self.backbone.layer1(x)
        out1 = self.exit1(x1)
        prob1 = torch.softmax(out1, dim=1)
        max_prob1, _ = torch.max(prob1, dim=1)
        
        # If the batch is confident enough, EXIT EARLY!
        # Do not calculate layer 2, 3, or 4.
        if (max_prob1 >= confidence_threshold).all():
            return out1, 1 

        # --- LAYER 2 ---
        x2 = self.backbone.layer2(x1)
        out2 = self.exit2(x2)
        prob2 = torch.softmax(out2, dim=1)
        max_prob2, _ = torch.max(prob2, dim=1)
        if (max_prob2 >= confidence_threshold).all():
            return out2, 2

        # --- LAYER 3 ---
        x3 = self.backbone.layer3(x2)
        out3 = self.exit3(x3)
        prob3 = torch.softmax(out3, dim=1)
        max_prob3, _ = torch.max(prob3, dim=1)
        if (max_prob3 >= confidence_threshold).all():
            return out3, 3

        # --- LAYER 4 (Final Exit) ---
        x4 = self.backbone.layer4(x3)
        out4 = self.exit4(x4)
        return out4, 4

if __name__ == "__main__":
    # Test script to verify the architecture
    print("Testing architecture...")
    model = EarlyExitResNet(num_classes=102)
    dummy_input = torch.randn(1, 3, 224, 224)
    logits, features = model(dummy_input)
    
    print("\nLogits (Predictions) shapes:")
    for i, l in enumerate(logits, 1):
        print(f"Exit {i}: {l.shape}")
        
    print("\nFeature shapes before exit pooling:")
    for i, f in enumerate(features, 1):
        print(f"Layer {i} output: {f.shape}")
    
    print("\nArchitecture test success!")
