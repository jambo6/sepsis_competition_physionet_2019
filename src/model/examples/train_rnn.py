from definitions import *
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from src.data.dataset import TimeSeriesDataset, ListDataset
from src.model.model_selection import stratified_kfold_cv
from src.model.nets import RNN

# Load the full dataset
dataset = TimeSeriesDataset().load(DATA_DIR + '/raw/data.tsd')
labels = load_pickle(DATA_DIR + '/processed/labels/utility_scores.pickle')

# For the RNN, lets only consider the variables that change often
data = dataset[['DBP', 'SBP', 'Temp', 'HR', 'MAP', 'ICULOS']]

# RNNs cannot deal with nans, so we will fill any nans in with zero
dataset.data[torch.isnan(dataset.data)] = 0

# Since this is an RNN, we will deal with the data as variable length lists
data = dataset.to_list()

# Get the id-indexed CV fold
cv, id_cv = stratified_kfold_cv(dataset, labels, n_splits=5, return_as_list=True, seed=1)

# Make train and test data
# TODO: This should really be train/test/val.
train_data = [data[i] for i in id_cv[0][0]]
train_labels = [labels[i] for i in cv[0][0]]
test_data = [data[i] for i in id_cv[0][1]]
test_labels = [labels[i] for i in cv[0][1]]

# Datasets
train_ds = ListDataset(train_data, train_labels)
test_ds = ListDataset(test_data, test_labels)

# Dataloaders. We use a batch size of 1 as we have lists not tensors.
train_dl = DataLoader(train_ds, batch_size=1)
test_dl = DataLoader(test_ds, batch_size=1)

# Model setup
model = RNN(in_channels=data[0].size(1), hidden_channels=10, out_channels=1)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

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


# Evaluate on test
# TODO: Sort this to work with threshold optimizer somehow
model.eval()
preds = []
with torch.no_grad():
    for data_ in test_data:
        preds.append(model(data_.unsqueeze(0)).view(-1))

    # Concat and evaluate
    full_predictions = torch.cat(preds)
    full_labels = torch.cat(test_labels)
    test_loss = loss_fn(full_predictions.view(-1), full_labels.view(-1))
    print('Test loss: {:.3f}'.format(test_loss))

