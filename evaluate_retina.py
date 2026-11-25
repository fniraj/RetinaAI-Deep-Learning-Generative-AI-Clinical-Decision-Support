import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

test_data = datasets.ImageFolder(
    "retina_split_dataset_v3/test", transform
)

test_loader = DataLoader(
    test_data, batch_size=8, shuffle=False
)

model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features, 4
)

model.load_state_dict(
    torch.load("retina_dr_model_v3.pth",
               map_location="cpu",
               weights_only=True)
)

model.eval()

actual = []
predicted = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(1)

        actual.extend(labels.tolist())
        predicted.extend(predictions.tolist())

correct = sum(a == p for a, p in zip(actual, predicted))

print(f"Test accuracy: {100 * correct / len(actual):.2f}%")

print("\nClassification Report:")
print(classification_report(
    actual, predicted,
    labels=[0, 1, 2, 3],
    target_names=test_data.classes,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predicted, labels=[0, 1, 2, 3]))