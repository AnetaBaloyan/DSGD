import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.functional import one_hot

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from dsgd.DSModel import DSModel

from pprint import pprint

# The Iris dataset
data = pd.read_csv("./data/iris.csv")

data = data[data.species != "versicolor"].reset_index(drop=True)
data = data.sample(frac=1).reset_index(drop=True)
cut = int(0.8 * len(data))

data = data.replace("virginica", 0).replace("setosa", 1)

model = DSModel(use_softmax=False)

cut = int(0.8 * len(data))

X_train = Tensor(data.iloc[:cut, :-1].values)
y_train = one_hot(Tensor(data.iloc[:cut, -1].values).long()).float()

X_test = data.iloc[cut:, :-1].values
y_test = one_hot(Tensor(data.iloc[cut:, -1].values).long()).float()

# Generating the rules for DSModel
model.generate_statistic_single_rules(X_train, breaks=2)

optimizer = torch.optim.Adam(model.masses, lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

model.train()

epoch = 0

loss_history = []
accuracy_history = []

checkpoints = []
loss_checkpoint = np.inf

# [optional] Measuring the time of the training process ONLY
start = time.time()

for epoch in range(200):
    try:
        optimizer.zero_grad()
        print("Processing Epoch %d:" % (epoch + 1))

        _train_var = Variable(Tensor(X_train), requires_grad=True)
        y_pred = model.forward(_train_var)

        loss = loss_fn(y_pred, Tensor(y_train))
        accuracy = accuracy_score(Tensor(y_train), torch.where(y_pred >= 0.5, 1, 0).detach())
        
        loss.backward()
        optimizer.step()
        
    except Exception as e:
        pprint("Last metrics are: " + str({
            "y_pred": y_pred,
            "loss": loss,
            "accuracy": accuracy,
            "epoch": epoch
        }))
        
        raise e
        
    # Saving the current state is loss decreased
    if loss.data.item() < loss_checkpoint:
        loss_checkpoint = loss
    
        curpath = os.path.dirname(__file__)
        filename = os.path.join(curpath, "checkpoints/checkpoint_epoch:%d.hdf5" % (epoch + 1))
        model.save_rules_bin(filename)

        print("[Checkpoint is saved into %s]" % filename)

    with torch.no_grad():
        print("\t Loss: ", loss.data.item())
        print("\t Accuracy: ", accuracy)

        accuracy_history.append(accuracy)
        loss_history.append(loss.data.item())

print("[THE EXECUTION TOOK %d SECONDS]" % (time.time() - start))

# Plot the learning curve
plt.plot(range(len(loss_history)), loss_history, label="Cross Entropy")
plt.plot(range(len(accuracy_history)), accuracy_history, label="Accuracy")

plt.legend()
plt.show()

# curpath = os.path.dirname(__file__)
# filename = os.path.join(curpath, "checkpoints")
# filename = os.path.join(filename, "checkpoint_epoch:151.hdf5")

# model = DSModel()
# model.load_rules_bin(filename)

# model.normalize()
# pprint(model.find_most_important_rules(threshold=0))

# y_pred = torch.where(model(Tensor(X_train)) >= 0.5, 1, 0)
# print("Confusion matrix for X_train")
# print(confusion_matrix(y_train.argmax(dim=-1), y_pred.argmax(dim=-1)))

# y_pred = torch.where(model(Tensor(X_test)) >= 0.5, 1, 0)
# print("Confusion matrix for X_test")
# print(confusion_matrix(y_test.argmax(dim=-1), y_pred.argmax(dim=-1)))
