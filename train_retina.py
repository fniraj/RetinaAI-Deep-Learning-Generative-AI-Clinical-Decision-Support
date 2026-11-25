import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

torch.manual_seed(42)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train_data = datasets.ImageFolder(
    "retina_split_dataset_v3/train", train_transform
)

val_data = datasets.ImageFolder(
    "retina_split_dataset_v3/validation", val_transform
)

train_loader = DataLoader(
    train_data, batch_size=8, shuffle=True
)

val_loader = DataLoader(
    val_data, batch_size=8
)

model = models.mobilenet_v3_large(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features, 4
)

# Load our Version 3 model
model.load_state_dict(
    torch.load(
        "retina_dr_model_v3.pth",
        map_location="cpu",
        weights_only=True
    )
)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Fine-tune the last six feature blocks
for param in model.features[-6:].parameters():
    param.requires_grad = True

for param in model.classifier[3].parameters():
    param.requires_grad = True

# Increase importance of Grade 3
counts = torch.bincount(torch.tensor(train_data.targets))
weights = len(train_data) / (4 * counts.float())
weights[3] *= 2.0

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam([
    {
        "params": model.features[-6:].parameters(),
        "lr": 0.00001
    },
    {
        "params": model.classifier[3].parameters(),
        "lr": 0.0001
    }
])

best_score = -1
best_accuracy = 0
best_grade3_recall = 0

for epoch in range(20):

    model.train()
    model.features.eval()

    total_loss = 0
    correct = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = 100 * correct / len(train_data)

    model.eval()
    correct = 0
    grade3_correct = 0
    grade3_total = 0

    with torch.no_grad():
        for images, labels in val_loader:

            outputs = model(images)
            predictions = outputs.argmax(1)

            correct += (predictions == labels).sum().item()

            grade3_mask = labels == 3

            grade3_total += grade3_mask.sum().item()

            grade3_correct += (
                (predictions == 3) & grade3_mask
            ).sum().item()

    val_acc = 100 * correct / len(val_data)

    grade3_recall = (
        100 * grade3_correct / max(grade3_total, 1)
    )

    # Prioritize overall accuracy and Grade 3 recall
    score = 0.7 * val_acc + 0.3 * grade3_recall

    print(
        f"Epoch {epoch + 1}/20 | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Train: {train_acc:.2f}% | "
        f"Validation: {val_acc:.2f}% | "
        f"Grade 3 Recall: {grade3_recall:.2f}%"
    )

    if score > best_score:

        best_score = score
        best_accuracy = val_acc
        best_grade3_recall = grade3_recall

        torch.save(
            model.state_dict(),
            "retina_dr_model_v4.pth"
        )

print("\nTraining completed.")
print(f"Selected validation accuracy: {best_accuracy:.2f}%")
print(f"Selected Grade 3 recall: {best_grade3_recall:.2f}%")