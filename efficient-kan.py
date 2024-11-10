from efficient_kan import KAN
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.metrics import *
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose(
    [transforms.Resize(size = (220,220)),
     transforms.RandomRotation(degrees=20),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5], std=[0.5])
    ]
)

train_path = 'Dataset/train'
train_dataset = datasets.ImageFolder(train_path, transform = transform_train)
trainloader = DataLoader(train_dataset, batch_size = 15, shuffle = True)

transform_test = transforms.Compose(
    [transforms.Resize(size = (220,220)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5], std=[0.5])
    ]
)

test_path = 'Dataset/test'
test_dataset = datasets.ImageFolder(test_path, transform = transform_test)
testloader = DataLoader(test_dataset, batch_size = 9, shuffle = True)

model = KAN([220*220*3, 100, 2])
model.to(device)

num_epoch = 35
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

train_losses = []
train_acc = []
correct = 0
total = 0

training_start_time = time.time()
for epoch in range(num_epoch):
    model.train()
    running_train_loss = 0.0

    for inputs, labels in trainloader:
        inputs = inputs.view(-1, 220*220*3).to(device)
        labels = labels.to(device)
    
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    train_loss = running_train_loss / len(trainloader)
    train_losses.append(train_loss)

    _, predicted = torch.max(outputs.data, 1)

    acc = accuracy_score(labels.cpu().detach().numpy(), predicted.cpu().detach().numpy())
    train_acc.append(acc)

    print(f"Época {epoch + 1}/{num_epoch} - Perda no treinamento: {train_loss:.6f} - Acurácia: {acc:.4f}")

training_time = time.time() - training_start_time
print(f"\nTempo total de treinamento: {training_time:.2f} segundos")

epochs = range(1, num_epoch + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'bo-')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'ro-')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.savefig('train_effcntkan.png', bbox_inches='tight')

model.eval()
with torch.no_grad():
        
    for images_test, labels_test in testloader:
        images_test = images_test.view(-1, 220*220*3).to(device)
        labels_test = labels_test.to(device)
        
        pred = model(images_test)
        _, predicted = torch.max(pred.data, 1)
        
print("!!!Teste finalizado!!!")

rotulos = labels_test.cpu().detach().numpy()
predito = predicted.cpu().detach().numpy()

acc = accuracy_score(rotulos, predito)
precision = precision_score(rotulos, predito, average='weighted')
recall = recall_score(rotulos, predito, average='weighted')
f1 = f1_score(rotulos, predito, average='weighted')

print(f"Acurácia: {acc * 100 :.2f}%")
print(f"Precisão: {precision * 100 :.2f}%")
print(f"Recall: {recall * 100 :.2f}%")
print(f"F1 Score: {f1 * 100 :.2f}%")

cm = confusion_matrix(rotulos, predito)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Anomalia', 'Normal'])
disp.plot(cmap=plt.cm.Blues)

plt.xlabel('Rótulo previsto')
plt.ylabel('Rótulo verdadeiro')
plt.savefig('test_effcntkan.png', bbox_inches='tight')