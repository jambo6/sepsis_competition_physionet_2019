from definitions import *
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from src.data.dataset import TimeSeriesDataset, ListDataset
from src.model.model_selection import stratified_kfold_cv
from src.model.nets import RNN
from src.model.optimizer import optimize_utility_threshold, compute_utility_from_indexes

# GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load the full dataset
dataset = TimeSeriesDataset().load(DATA_DIR + '/raw/data.tsd')
labels = torch.Tensor(load_pickle(DATA_DIR + '/processed/labels/utility_scores.pickle'))

# For the RNN, lets only consider the variables that change often
data = dataset[['DBP', 'SBP', 'Temp', 'HR', 'MAP', 'ICULOS']]

# RNNs cannot deal with nans, so we will fill any nans in with zero
dataset.data[torch.isnan(dataset.data)] = 0

# Since this is an RNN, we will deal with the data as variable length lists
data = dataset.to_list()

# Get the id-indexed CV fold. We need both patient indexes and time index.
cv, id_cv = stratified_kfold_cv(dataset, labels, n_splits=5, return_as_list=True, seed=1)
train_idxs, test_idxs = cv[0]
train_id_idxs, test_id_idxs = id_cv[0]

# Make train and test data
# TODO: This should really be train/test/val.
train_data = [data[i].to(device) for i in train_id_idxs]
train_labels = [labels[i].to(device) for i in train_idxs]
test_data = [data[i].to(device) for i in test_id_idxs]
test_labels = [labels[i].to(device) for i in test_idxs]

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

    if epoch % print_freq == 0:
        train_loss = np.mean(train_losses)
        print("Epoch: {:.3f}  Average training loss: {:.3f}".format(epoch, train_loss))


# Evaluate on test
model.eval()
train_preds, test_preds = [], []
with torch.no_grad():
    # Predict train
    for batch in train_data:
        train_preds.append(model(batch.unsqueeze(0)).view(-1))

    # Predict test
    for batch in test_data:
        test_preds.append(model(batch.unsqueeze(0)).view(-1))

# Concat
train_preds = torch.cat(train_preds).view(-1).detach()
test_preds = torch.cat(test_preds).view(-1).detach()
train_labels = torch.cat(train_labels).view(-1).detach()
test_labels = torch.cat(test_labels).view(-1).detach()

# Compute losses
train_loss = loss_fn(train_labels, train_preds)
test_loss = loss_fn(test_labels, test_preds)
print('Train loss: {:.3f}'.format(train_loss))
print('Test loss: {:.3f}'.format(test_loss))

# Compute the score on the utility function
train_idxs, test_idxs = torch.cat(train_idxs), torch.cat(test_idxs)
tfm_np = lambda x: x.cpu().numpy()
train_preds, test_preds = tfm_np(train_preds), tfm_np(test_preds)
thresh = optimize_utility_threshold(train_preds, idxs=train_idxs)
train_utility = compute_utility_from_indexes(train_preds, thresh, idxs=train_idxs)
test_utility = compute_utility_from_indexes(test_preds, thresh, idxs=test_idxs)
print('Train utility score: {:.3f}'.format(train_utility))
print('Test utility score: {:.3f}'.format(test_utility))
