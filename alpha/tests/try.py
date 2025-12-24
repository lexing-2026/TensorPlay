from alpha.tests import Tensor, Module, Dense, Conv2D, BatchNorm
import numpy as np

class Model(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.C1 = Conv2D(3, 8, 3, 2, padding='same')
        self.B1 = BatchNorm(8)
        self.C2 = Conv2D(8, 16, 3, 2, padding='same')
        self.C3 = Conv2D(16, 32, 3, 2, padding='same')
        self.F2 = Dense(32 * 16* 16, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.C1(x)
        x = x.relu()
        x = self.B1(x)
        x = self.C2(x)
        x = x.relu()
        x = self.C3(x)
        x = x.relu()
        x = x.flatten()
        x = self.F2(x)
        return x.softmax()


if __name__ == '__main__':
    model = Model(num_classes=2)
    x = Tensor(np.random.randn(10, 128, 128, 3))
    y = model(x)
    assert y.shape == (10, 2)
