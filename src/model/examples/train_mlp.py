"""
train_mlp.py
==========================
This example is very similar to the LGBM example, except we use a multi-layer pe
"""
from definitions import *
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from src.data.dataset import TimeSeriesDataset
from src.data.functions import torch_ffill
from src.model.model_selection import stratified_kfold_cv
from src.model.nets import MLP
from src.model.optimizer import optimize_utility_threshold, compute_utility_from_indexes

# GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the dataset
dataset = TimeSeriesDataset().load(DATA_DIR + '/raw/data.tsd')

# Load the training labels
labels = torch.Tensor(load_pickle(DATA_DIR + '/processed/labels/utility_scores.pickle'))

# Apply a forward fill
dataset.data = torch_ffill(dataset.data)

# MLPs cannot deal with nans, so we will fill any nans in with zero
dataset.data[torch.isnan(dataset.data)] = 0

# Extract features here, like in the lgbm example

# Cross val
cv, _ = stratified_kfold_cv(dataset, labels, n_splits=5, seed=1)

# Extract machine learning data
X = dataset.to_ml()
assert len(X) == len(labels)    # Sanity check
X, labels = X.to(device), labels.to(device)

# Train test
train_idxs, test_idxs = cv[0]
train_data, train_labels = X[train_idxs], labels[train_idxs]
test_data, test_labels = X[test_idxs], labels[test_idxs]

# Setup dataloader
train_ds = TensorDataset(train_data, train_labels)
train_dl = DataLoader(train_ds, batch_size=64)
test_ds = TensorDataset(test_data, test_labels)
test_dl = DataLoader(test_ds, batch_size=128, shuffle=False)

# Model setup
model = MLP(in_channels=X.shape[1], hidden_channels=10, out_channels=1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
model.to(device)

# Training loop
n_epochs = 1
print_freq = 1
model.train()
for epoch in range(n_epochs):
    train_losses = []
    for i, batch in tqdm(enumerate(train_dl)):
        optimizer.zero_grad()
        inputs, true = batch
        outputs = model(inputs)
        loss = loss_fn(outputs.view(-1), true.view(-1))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if i > 10:
            break

    if epoch % print_freq == 0:
        train_loss = np.mean(train_losses)
        print("Epoch: {:.3f}  Average training loss: {:.3f}".format(epoch, train_loss))


# Evaluation
model.eval()
with torch.no_grad():
    train_preds = model(train_ds.tensors[0]).view(-1).detach()
    test_preds = model(test_ds.tensors[0]).view(-1).detach()

    train_loss = loss_fn(train_preds, train_labels.view(-1))
    test_loss = loss_fn(test_preds, test_labels.view(-1))

print('Train loss: {:.3f}'.format(train_loss))
print('Test loss: {:.3f}'.format(test_loss))

tfm_np = lambda x: x.cpu().numpy()
train_preds, test_preds = tfm_np(train_preds), tfm_np(test_preds)
thresh = optimize_utility_threshold(train_preds, idxs=train_idxs)
train_utility = compute_utility_from_indexes(train_preds, thresh, idxs=train_idxs)
test_utility = compute_utility_from_indexes(test_preds, thresh, idxs=test_idxs)
print('Train utility score: {:.3f}'.format(train_utility))
print('Test utility score: {:.3f}'.format(test_utility))

