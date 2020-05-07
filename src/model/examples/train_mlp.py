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
from src.data.dataset import TimeSeriesDataset
from src.data.functions import torch_ffill
from src.model.model_selection import stratified_kfold_cv
from src.model.nets import MLP

# Load the dataset
dataset = TimeSeriesDataset().load(DATA_DIR + '/raw/data.tsd')

# Load the training labels
labels = load_pickle(DATA_DIR + '/processed/labels/utility_scores.pickle')

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

# Train test
train_data, train_labels = X[cv[0][0]], labels[cv[0][0]]
test_data, test_labels = X[cv[0][1]], labels[cv[0][1]]

# Setup dataloader
train_ds = TensorDataset(train_data, train_labels)
train_dl = DataLoader(train_ds, batch_size=64)

# Model setup
model = MLP(in_channels=X.shape[1], hidden_channels=10, out_channels=1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
n_epochs = 1
print_freq = 1
model.train()
for epoch in range(n_epochs):
    train_losses = []
    for i, batch in enumerate(train_dl):
        optimizer.zero_grad()
        inputs, true = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, true)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if i > 10:
            break

    if epoch % print_freq == 0:
        train_loss = np.mean(train_losses)
        print("Epoch: {:.3f}  Average training loss: {:.3f}".format(epoch, train_loss))

# Evaluation
# TODO: Sort this to work with threshold optimizer somehow
model.eval()
with torch.no_grad():
    pred_test = model(test_data)
print('Test loss: {:.3f}'.format(loss_fn(pred_test, test_labels)))

