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


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


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


def give_parameter_set(argument_int):
    betas = [0.5, 0.7, 0.9]
    num_hidden_neurons = [50, 100,  200, 400]
    time_windows  = [500, 1000, 2000]
    loss_function_wrong_classifications = [[0.8, 0,2], [0.95, 0.05]]

    parameter_set = []
    for beta in betas:
        for num_hidden in num_hidden_neurons:
            for time_window in time_windows:
                for loss_function_ratio in loss_function_wrong_classifications:
                    parameter_set.append([beta, num_hidden, time_window, loss_function_ratio])
    
    parameters = parameter_set[argument_int]
        #beta, hidden, timewindow, loss ratio
    return parameters[0], parameters[1], parameters[2], parameters[3]

# beta = [0.5, 0.7, 0.9]
# number_hidden = [100,  200, 400, 800]
# time_window  = [500, 1000, 2000]
# loss_function_wrong_classification = [[0.8, 0,2], [0.95, 0.05]]


######## MODEL SETTINGS ##############
job_id = int(sys.argv[1]) #TODO here we need the int from habrok
beta, hidden_neurons, timewindow, loss_ratio = give_parameter_set(job_id) 

print(beta, hidden_neurons, timewindow, loss_ratio)

# K_fold settings and result variables
k_folds = 2
results = {}
kfold = KFold(n_splits=k_folds, shuffle=True)

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

num_epochs = 4


#set true if you want to see print
verbose = True

# K-fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(kfold.split(trainset)):


    print(f'FOLD {fold}')
    print('--------------------------------')
    
    while True:
        try:
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            #TODO: I disabled the shuffle option as this is mutually exclusive with the k-means samples option. Check whether this is okay with the person that implemented the shuffle option.
            trainloader = make_dataloader(trainset, batch_size, samplerset=train_subsampler)
            testloader = make_dataloader(trainset, batch_size, samplerset=test_subsampler)
        except:
            time.sleep(5)
            continue
        break


    loss_hist = []
    train_acc_hist = []

    # TODO: Check whether the weights are correctly reset. And the net is cleared indeed
    net.apply(reset_weights)

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
        #validation of k folds
        val_acc_hist = []
        for i, (data, targets) in enumerate(iter(trainloader)):

            data = data.to(device)
            targets = targets.to(device)

            spk_rec = forward_pass(net, data)
            loss_val = loss_fn(spk_rec, targets) #calculate the loss value
            # Store loss history for future plotting
            acc = SF.accuracy_rate(spk_rec, targets)
            val_acc_hist.append(acc)


            if verbose:
                print(f"Epoch {epoch}, Iteration {i} \n acc: {acc:.2f}", flush = True)
        validation_acc.append(np.mean(val_acc_hist))
        print(f"number of batches in the test set{i}")


    #TODO: Implement and expand which parts of the k-th fold model should be stored.
    #TODO: Implement k-validation accuracy calculation and append it to the results per k-fold.
    results[fold] = validation_acc

    import matplotlib.pyplot as plt

    # Plot Loss
    # fig = plt.figure(facecolor="w")
    # plt.plot(validation_acc)
    # plt.title("Test Set Accuracy")
    # plt.xlabel("Iteration")
    # plt.ylabel("Accuracy")
    # plt.show()


# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')




sum = np.zeros(num_epochs)
for key, value in results.items():
    sum += np.array(value)
sum /= k_folds
if verbose:
    print(f'Average: {sum[-1] * 100:.2f} %')


results = [beta, hidden_neurons, timewindow, loss_ratio, sum]

results = pd.DataFrame(results)

results.to_csv("results_" + str(job_id) + ".csv")

#parameters to tune
# beta = [0.5, 0.7, 0.9]
# number_hidden = [50, 100,  200, 400]
# time_window  = [500, 1000, 2000]
# loss_function_wrong_classification = [[0.8, 0,2], [0.95, 0.05]]


