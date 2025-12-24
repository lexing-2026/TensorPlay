import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

def benchmark():
    device = torch.device("cpu")

    print("="*50)
    print(f"PyTorch Benchmark: LeNet-5 on MNIST ({device})")
    print("="*50) 

    # 1. Data Setup
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    # Download=True might be redundant if already downloaded by tensorplay but good for safety
    train_dataset = datasets.MNIST(root='./data_torch', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data_torch', train=False, download=True, transform=transform)
    
    batch_size = 32
    # num_workers=0 to match simple loader likely used in TensorPlay (or single threaded)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Model Setup
    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = LeNet().to(device)
    
    # 3. Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop
    epochs = 2
    total_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100 * correct / total:.2f}%"
                })
        
        epoch_duration = time.time() - epoch_start_time
        epoch_acc = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        
        print(f"Epoch {epoch+1} Summary: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, Time: {epoch_duration:.2f}s")

    total_training_time = time.time() - total_start_time
    print(f"Total Training Time: {total_training_time:.2f}s")

    # 5. Testing
    print("Starting evaluation...")
    model.eval()
    correct = 0
    total = 0
    test_start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    test_duration = time.time() - test_start_time
    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Time: {test_duration:.2f}s")
    print("="*50)

if __name__ == '__main__':
    benchmark()
