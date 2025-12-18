# Best model

from cProfile import label
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class Classifier(nn.Module):
   
    def __init__(self, num_classes=10,freeze_backbone=True): 
        super().__init__()
        
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

        # Replace last layer for 10 classes
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # Freeze backbone if required
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class MyClassifier():
    
    
    def __init__(self):
        self.class_labels = [
            'edible_1','edible_2','edible_3','edible_4','edible_5',
            'poisonous_1','poisonous_2','poisonous_3','poisonous_4','poisonous_5'
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Preprocessing pipeline
        
        imagenet_means = (0.485, 0.456, 0.406)
        imagenet_stds = (0.229, 0.224, 0.225)
        
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_means, imagenet_stds),
            
        ])

        self.classifier = None

    def setup(self, checkpoint_path=None, freeze_backbone=True):
        self.classifier = Classifier(
            num_classes=len(self.class_labels),
            freeze_backbone=freeze_backbone
        ).to(self.device)

        if checkpoint_path is not None:
            self.classifier.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )

        self.classifier.eval()

    def test_image(self, image):
        
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.classifier(x)
            pred = torch.argmax(outputs, dim=1).item()
        return self.class_labels[pred]

    def test_image_calibrated(self, image):
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.classifier(x)
            pred = torch.argmax(outputs, dim=1).item()
            label = self.class_labels[pred]

        return label.startswith("poisonous")
    
