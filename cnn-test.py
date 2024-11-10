import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_test = transforms.Compose(
    [transforms.Resize(size = (220,220)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5], std=[0.5])
    ]
)

test_path = 'Dataset/test'
test_dataset = datasets.ImageFolder(test_path, transform = transform_test)
testloader = DataLoader(test_dataset, batch_size = 9, shuffle = True)

model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),  
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
    nn.Flatten(),
    nn.Linear(64 * 110 * 110, 100),
    nn.Dropout(p=0.3),
    nn.ReLU(),
    nn.Linear(100, 2)
)

model.to(device)
model.load_state_dict(torch.load('termo_model.pth', weights_only=True))
model.eval()

with torch.no_grad():
        
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
print("!!!Teste finalizado!!!")

rotulos = labels.cpu().detach().numpy()
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
plt.savefig('test_cnn.png', bbox_inches='tight')