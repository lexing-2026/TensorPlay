from alpha.TensorPlay import DataLoader, Dense, Module, Adam, accuracy, mse, load_moons

class MoonClassify(Module):
    def __init__(self):
        super().__init__()
        self.F1 = Dense(2, 16)
        self.F2 = Dense(16, 1)

    def forward(self, x):
        x = self.F1(x)
        x = x.relu()
        x = self.F2(x)
        return x

train_dataset = DataLoader(load_moons())
model = MoonClassify()
optimizer = Adam(lr=0.001)

for epoch in range(1000):
    for x, y in train_dataset:
        pred = model(x)
        loss = mse(y, pred)
        if epoch % 100 == 0:
            print(accuracy(pred, y))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
