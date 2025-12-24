import tensorplay as tp
import tensorplay.optim as optim
from tensorplay.optim import lr_scheduler

def test_optimizers():
    print("Testing Optimizers...")
    params = [tp.Tensor([1.0], requires_grad=True)]
    
    # Test SGD
    print("Testing SGD...")
    opt = optim.SGD(params, lr=0.1)
    opt.zero_grad()
    loss = params[0] * 2
    loss.backward()
    opt.step()
    # grad = 2, param = 1 - 0.1 * 2 = 0.8
    assert abs(params[0].item() - 0.8) < 1e-5
    print("SGD passed")

    # Test Adam
    print("Testing Adam...")
    params = [tp.Tensor([1.0], requires_grad=True)]
    opt = optim.Adam(params, lr=0.1)
    opt.zero_grad()
    loss = params[0] * 2
    loss.backward()
    opt.step()
    print("Adam passed")

    # Test AdamW
    print("Testing AdamW...")
    params = [tp.Tensor([1.0], requires_grad=True)]
    opt = optim.AdamW(params, lr=0.1)
    opt.zero_grad()
    loss = params[0] * 2
    loss.backward()
    opt.step()
    print("AdamW passed")

    # Test RMSprop
    print("Testing RMSprop...")
    params = [tp.Tensor([1.0], requires_grad=True)]
    opt = optim.RMSprop(params, lr=0.1)
    opt.zero_grad()
    loss = params[0] * 2
    loss.backward()
    opt.step()
    print("RMSprop passed")

    # Test Adagrad
    print("Testing Adagrad...")
    params = [tp.Tensor([1.0], requires_grad=True)]
    opt = optim.Adagrad(params, lr=0.1)
    opt.zero_grad()
    loss = params[0] * 2
    loss.backward()
    opt.step()
    print("Adagrad passed")

def test_schedulers():
    print("\nTesting Schedulers...")
    params = [tp.Tensor([1.0], requires_grad=True)]
    opt = optim.SGD(params, lr=0.1)
    
    # StepLR
    print("Testing StepLR...")
    scheduler = lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
    print(f"Initial LR: {scheduler.get_last_lr()}")
    scheduler.step()
    print(f"Step 1 LR: {scheduler.get_last_lr()}")
    assert abs(scheduler.get_last_lr()[0] - 0.01) < 1e-5
    print("StepLR passed")

if __name__ == "__main__":
    test_optimizers()
    test_schedulers()
