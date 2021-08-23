import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from gossipy.data import DataHandler
from gossipy.model.nn import TorchMLP


def torch_mlp(data_handler: DataHandler,
              n_epochs: int=300,
              batch_size: int=16,
              learning_rate: float=0.01,
              l2_reg: float=0.001,
              verbose: bool=True) -> float:

    class DataSampler():
        def __init__(self,
                    X_tensor: torch.FloatTensor,
                    y_tensor: torch.LongTensor,
                    batch_size=1,
                    shuffle=True):
            self.X_tensor = X_tensor
            self.y_tensor = y_tensor
            self.batch_size = batch_size
            self.shuffle = shuffle

        def __len__(self):
            return int(np.ceil(self.X_tensor.shape[0] / self.batch_size))

        def __iter__(self):
            n = self.X_tensor.shape[0]
            idxlist = list(range(n))
            if self.shuffle: np.random.shuffle(idxlist)
            for _, start_idx in enumerate(range(0, n, self.batch_size)):
                end_idx = min(start_idx + self.batch_size, n)
                data_tr = self.X_tensor[idxlist[start_idx:end_idx]]
                data_te = self.y_tensor[idxlist[start_idx:end_idx]]
                yield data_tr, data_te

    mlp = TorchMLP(data_handler.Xtr.shape[1])
    optimizer = torch.optim.SGD(mlp.parameters(),
                                lr=learning_rate,
                                weight_decay=l2_reg)
    loss_function = F.mse_loss
    trainloader = DataSampler(data_handler.Xtr,
                              data_handler.ytr,
                              batch_size=batch_size)
    for epoch in range(n_epochs):
        current_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            optimizer.zero_grad()
            outputs = mlp(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
        if verbose and epoch % 100 == 99:
            print('Loss after epoch %5d: %.3f' % (epoch + 1, current_loss/(i+1)))
        current_loss = 0

    y_pred = mlp(data_handler.Xte)
    #y_pred = (y_pred.detach().cpu().numpy() >= 0.5)
    y_pred = y_pred.detach().cpu().numpy()
    y_true = data_handler.yte.detach().cpu().numpy()
    #return accuracy_score(y_true.flatten(), y_pred).astype(float)
    te0 = data_handler.te_fmap[0]
    te1 = data_handler.te_fmap[1]
    return roc_auc_score(y_true.flatten(), y_pred).astype(float), \
           roc_auc_score(y_true[te0].flatten(), y_pred[te0]).astype(float), \
           roc_auc_score(y_true[te1].flatten(), y_pred[te1]).astype(float)


def sklearn_mlp(data_handler: DataHandler,
                n_epochs: int=300,
                batch_size: int=16,
                learning_rate: float=0.01,
                l2_reg: float=0.001,
                verbose: bool=True) -> float:

    clf = MLPClassifier(max_iter=n_epochs,
                        learning_rate_init=learning_rate,
                        alpha=l2_reg,
                        batch_size=batch_size,
                        verbose=verbose).fit(data_handler.Xtr, data_handler.ytr.flatten())
    #return accuracy_score(data_handler.yte, clf.predict(data_handler.Xte)).astype(float)
    te0 = data_handler.te_fmap[0]
    te1 = data_handler.te_fmap[1]
    return accuracy_score(data_handler.yte, clf.predict(data_handler.Xte)).astype(float), \
        roc_auc_score(data_handler.yte[te0], clf.predict_proba(data_handler.Xte[te0])[:, 1]).astype(float), \
        roc_auc_score(data_handler.yte[te1], clf.predict_proba(data_handler.Xte[te1])[:, 1]).astype(float)

if __name__ == "__main__":
    mlp = TorchMLP(10)
    for name, param in mlp.named_parameters():
        print(name.split('.')[0])
        print(getattr(mlp, name.split('.')[0]))