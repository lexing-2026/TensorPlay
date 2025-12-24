import tensorplay as tp
import tensorplay.nn as nn
import tensorplay.optim as optim
from tensorplay.vision import datasets, transforms
from tensorplay.utils.data import DataLoader
import time
from tqdm import tqdm

print(f"TensorPlay: {tp.__file__}")

def benchmark():

    # 1. Data Setup
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_dataset = datasets.MNIST(train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(train=False, download=True, transform=transform)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
            x = self.pool(self.conv1(x).relu())
            x = self.pool(self.conv2(x).relu())
            x = x.view(-1, 16 * 5 * 5)
            x = self.fc1(x).relu()
            x = self.fc2(x).relu()
            x = self.fc3(x)
            return x

    model = LeNet()
    device = tp.device("cpu")
    model.to(device)

    print("="*50)
    print(f"TensorPlay Benchmark: LeNet-5 on MNIST ({device})")
    print("="*50)
    
    # 3. Optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop
    epochs = 2  # Increased to 5 to check convergence
    total_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        
        # Use simple iteration for cleaner timing without tqdm overhead in inner loop if needed, 
        # but tqdm is good for progress visibility.
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, (inputs, labels) in enumerate(pbar):
            # print(f"Batch {i}")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = outputs.data.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).float().sum().item()
            
            if i % 50 == 0:
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
    
    with tp.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = outputs.data.argmax(1)
            total += labels.size(0)
            correct += (predicted == labels).float().sum().item()
            
    test_duration = time.time() - test_start_time
    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Time: {test_duration:.2f}s")
    print("="*50)

if __name__ == '__main__':
    benchmark()
