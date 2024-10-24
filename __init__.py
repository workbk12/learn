import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision as tv
import time


def main():
    BATCH_SIZE=256

    train_ds=tv.datasets.FashionMNIST('.', train=True, transform=tv.transforms.ToTensor(), download=True)
    test_ds=tv.datasets.FashionMNIST('.', train=False, transform=tv.transforms.ToTensor(), download=True)

    train=DataLoader(train_ds, batch_size=BATCH_SIZE)
    test=DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    model=torch.nn.Sequential(
        torch.nn.Flatten(),  
        torch.nn.Linear(784, 1500),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1500, 600),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(600, 200),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(200, 10)
    )

    loss=torch.nn.CrossEntropyLoss()
    trainer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs=20

    def train_model():
        for ep in range(num_epochs):
            start = time.time()
            train_iters, train_passed=0, 0
            train_loss, train_ass=0., 0.

            model.train()
            for X, y in train:
                trainer.zero_grad()
                y_pred=model(X)
                l=loss(y_pred, y)
                l.backward()
                trainer.step()
                train_loss+=l.item()
                train_ass+=(y_pred.argmax(dim=1)==y).sum().item()
                train_iters+=1
                train_passed+=len(X)

            test_iters, test_passed = 0, 0
            test_loss, test_ass = 0., 0.

            model.eval()
            for X, y in test:
                y_pred = model(X)
                l = loss(y_pred, y)
                test_loss += l.item()
                test_ass += (y_pred.argmax(dim=1) == y).sum().item()
                test_iters += 1
                test_passed += len(X)

            print('ep: {}, taked: {}, train_loss: {}, train_ass: {}, test_loss: {}, test_ass: {}'
                  .format(ep, time.time()-start, train_loss/train_iters, train_ass/train_passed, test_loss/test_iters, test_ass/test_passed))

    train_model()

if __name__ == '__main__':
    main()