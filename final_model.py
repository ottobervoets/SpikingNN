import tonic
import tonic.transforms as transforms
import torch
from snntorch import utils

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt
from snntorch import utils
import torch.nn as nn

from torch.utils.data import DataLoader
from tonic import DiskCachedDataset
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import sys
import time


def download_data(verbose = False, time_window = 1000):
    sensor_size = tonic.datasets.NMNIST.sensor_size

    # Denoise removes isolated, one-off events
    # time_window
    frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                        transforms.ToFrame(sensor_size=sensor_size,
                                                            time_window=time_window)
                                        ])

    trainset = tonic.datasets.NMNIST(save_to='./tmp/data', transform=frame_transform, train=True)
    #  Reduce size of MNIST training set

    testset = tonic.datasets.NMNIST(save_to='./tmp/data', transform=frame_transform, train=False)
    if verbose:
        print(f"Number of observations in train set = {len(trainset)}\n")
        print("Number of observations in test set", len(testset))
    #  Reduce size of MNIST training set
    return trainset, testset



def make_dataloader(data_set, batch_size, samplerset, shuffel = False):
    cached_trainset = DiskCachedDataset(data_set, cache_path='./cache/nmnist/train')
    cached_dataloader = DataLoader(cached_trainset)
    return DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=shuffel, sampler=samplerset)

#Overwrite forward pass
def forward_pass(net, data):
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(data.size(0)):  # data.size(0) = number of time steps
      spk_out, mem_out = net(data[step])
      spk_rec.append(spk_out)

  return torch.stack(spk_rec)

# If true, it loads the saved model from the repository
LOAD_MODEL = False
TRAIN_MODEL = True
model_path = "trained_snn.pt"


beta = 0.9
hidden_neurons = 800
timewindow = 2000
loss_ratio = [0.8, 0.2]


batch_size = 128

trainset, testset = download_data(time_window = timewindow) #donwload mnist data


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# neuron and simulation parameters
spike_grad = surrogate.atan()

input_neurons = 34 * 34 * 2  # image size 34 * 34 times polarity * 2
output_neurons = 10  # 10 classes/labels

#define layers
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(input_neurons, hidden_neurons),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Linear(hidden_neurons, output_neurons),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

#optimizer and loss function
optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999), weight_decay = 1e-5)
loss_fn = SF.mse_count_loss(correct_rate=loss_ratio[0], incorrect_rate=loss_ratio[1])   #We want 80 percet of spikes from the correct classes and 20% from incorrect classes, in order to avoid dead neurons

num_epochs = 15


# If requested, load a model.
if LOAD_MODEL:
    net.load_state_dict(torch.load(model_path))



#set true if you want to see print
verbose = True


# Load the data
while True:
    try:
        trainloader = DataLoader(trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))
    except:
        time.sleep(5)
        continue
    break


loss_hist = []
train_acc_hist = []


if TRAIN_MODEL:
    print("start training")
    # training loop
    validation_acc = []
    for epoch in range(num_epochs):
        #Train the network
        for i, (data, targets) in enumerate(iter(trainloader)):
            
            data = data.to(device)
            targets = targets.to(device)

            net.train()
            spk_rec = forward_pass(net, data)
            loss_val = loss_fn(spk_rec, targets) #calculate the loss value
            # Gradient calculation + weight update
            optimizer.zero_grad() #clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
            loss_val.backward() #computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
            optimizer.step() #causes the optimizer to take a step based on the gradients of the parameters.

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            acc = SF.accuracy_rate(spk_rec, targets)
            train_acc_hist.append(acc)


            if verbose:
                print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}", flush = True)
                print(f"Accuracy: {acc * 100:.2f}%\n")


test_acc_hist = []
#Test the test-dataset
print("start test procedure")
for i, (data, targets) in enumerate(iter(testloader)):

    data = data.to(device)
    targets = targets.to(device)

    spk_rec = forward_pass(net, data)
    loss_val = loss_fn(spk_rec, targets) #calculate the loss value
    # Store loss history for future plotting
    acc = SF.accuracy_rate(spk_rec, targets)
    test_acc_hist.append(acc)

print("mean accuracy", np.mean(test_acc_hist))

# Save the model if you didn't load a saved model.
if not LOAD_MODEL:
    torch.save(net.state_dict(), model_path)