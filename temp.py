import torch
import numpy as np
import matplotlib.pyplot as plt


class MyDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        return self
    def __next__(self):
        X_orig = np.random.randint(0, 100)
        X = (X_orig - 50.0) / (100/np.sqrt(12))
        if X_orig < 50:
            y = X * np.random.randint(1, 3) + np.random.uniform(0, 2)
        else:
            y = np.sin(np.pi * X) + np.random.uniform(-0.5, 0.5)
        return np.array([X], dtype=np.float32), np.array([y], dtype=np.float32)


class TwoLayerModel(torch.nn.Module):
    def  __init__(self, D_IN, D_OUT, H):

        super(TwoLayerModel, self).__init__()
        self._l1 = torch.nn.Linear(D_IN, H)
        self._l2 = torch.nn.Linear(H, H)
        self._l3 = torch.nn.Linear(H, D_OUT)

    def forward(self, x):
        x = self._l1(x).clamp(min=0)
        x = self._l2(x).clamp(min=0)
        x = self._l3(x)
        return x


if __name__ == "__main__":

    train_dataset = MyDataset()
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 8,
        num_workers = 2,
    )

    model = TwoLayerModel(
        D_IN = 1,
        D_OUT = 1,
        H = 4,
    )

    loss_fn = torch.nn.MSELoss(reduction="mean")
    optim = torch.optim.Adam(model.parameters())

    iter_train_dataloader = iter(train_dataloader)

    for i in range(10_000):
        optim.zero_grad()
        X, y = next(iter_train_dataloader)
        output = model(X)
        loss = loss_fn(output, y)
        if (i+1) % 1_000 == 0:
            print(f"loss @ {i+1}th step: {loss}")
        loss.backward()
        optim.step()


    X, y, y_pred = [], [], []
    iter_dataset = iter(train_dataset)
    for _ in range(100):
        _X, _y = next(iter_dataset)
        X.append(_X)
        y.append(_y)
        with torch.no_grad():
            y_pred.append(model(torch.Tensor(_X)))


    X, y, y_pred = zip(*sorted(zip(X, y, y_pred)))

    fig, ax = plt.subplots(1, 1)
    ax.set_title("Predicted vs Actual")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.scatter(np.stack(X), np.stack(y), color="tab:blue")
    ax.plot(np.stack(X), np.stack(y_pred), linewidth=3, color="tab:orange")
    plt.show()
