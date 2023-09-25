# Data Science Project
# Duncan Torbet
# 07/09/2023

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import csv

## This is just to measure the execution time of the file.
start = time.time()

df = pd.read_csv("D:/Users/dunca/Desktop/2023/Modules/DS/Project/higgs-boson/training/training.csv") # Constructing the DataFrame.
print(df)


class NeuralNet(nn.Module): # This is the Neural Net.

    def __init__(self):
        super().__init__() # Here we initialize our super class containing our NN:
        self.M1 = nn.Linear(30, 90) # Initial Layer
        self.M2 = nn.Linear(90, 180) # Hidden Layer 1
        self.M3 = nn.Linear(180, 180) # Hidden Layer 2
        self.M4 = nn.Linear(180, 180) # Hidden Layer 3
        self.M5 = nn.Linear(180, 90) # Hidden Layer 4
        self.M6 = nn.Linear(90, 1) # Ouput Layer
        self.R = nn.ReLU() # Here we utilize the Rectified Linear Unit activation fn.
        self.sigmoid = nn.Sigmoid() # Our final parameters need to be between 0 and 1, thus we utilize the sigmoid.

    def forward(self, x): # Here we define our forward propagation:
        x = self.R(self.M1(x))
        x = self.R(self.M2(x))
        x = self.R(self.M3(x))
        x = self.R(self.M4(x))
        x = self.R(self.M5(x))
        x = self.sigmoid(self.M6(x))
        return x # Done.


def ModelTraining(x, y, f, xval, yval, nEpochs): # Here is how we train our model.
    opt = torch.optim.AdamW(f.parameters(), lr=0.0008)
    opt_val = torch.optim.AdamW(f.parameters(), lr=0.0008)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma = 0.99)
    scheduler_val = torch.optim.lr_scheduler.ExponentialLR(opt_val, gamma = 0.99)

    tolerance = 1e-5
    L = nn.BCELoss() # Here we have our loss function which is the Binary Cross Entropy function. (binary because our output is between 0 and 1)
    L_val = nn.BCELoss()

    # Training the model
    losses = []
    losses_validation = []
    KILL_SWITCH = 0
    epoch_tol = 30

    for i in range(nEpochs): # This will iterate through the Neural Network a bunch of times (epochs)
        opt.zero_grad() # We are make our gradient zero for each epoch.
        loss_value = L(f(x), y) # We determine our loss value.
        loss_value.backward()
        opt.step()
        losses.append(loss_value.item())
        scheduler.step()

        opt_val.zero_grad()
        loss_value_validation = L_val(f(xval), yval) # We determine our loss value.
        loss_value_validation.backward()
        opt_val.step()
        losses_validation.append(loss_value_validation.item())
        scheduler_val.step()

        diff = np.abs(losses[i]-losses[i-1])
        diff_val = np.abs(losses_validation[i]-losses_validation[i-1])

        if diff <= tolerance and diff_val <= tolerance:
            KILL_SWITCH += 1
        if KILL_SWITCH >= epoch_tol:
            print("Stopped early; tolerance reached.")
            break
        else:
            pass

    return f, losses, losses_validation


def AMS(s, b, b_reg): # Here we have the Average Mean Significance.

    return np.sqrt(2 * ((s + b + b_reg)*np.log(1 + s/(b+b_reg)) - s))


labels = df.columns[1:-2] # Dataframe headings.
tensor = torch.tensor(df[labels].values.astype(float)).float() # Here we have our tensor

signal_background = []
for i in df["Label"].values:
    if i == "s":
        signal_background.append([1]) # Denoting signal events as 1
    else:
        signal_background.append([0]) # Denoting background events as 0

output = torch.tensor(signal_background).float() # Making a tensor of the events.

Entries = len(output)

# 0th entry is eventID, 30 entries of info, + 1 weight entry, last entry is sig or back.
## Now, we need a training set, test set and validation set. Let's use 7:2:1.

#####################################################
################## TRAINING #########################
#####################################################

train_range = int(0.7*Entries)
validation_range = int(0.1*Entries) + train_range
test_range = Entries ## NOTE THAT THESE ARE CUMULATIVE


f = NeuralNet() # Initializing the Neural Net.
f, losses, losses_validation = ModelTraining(tensor[:train_range], output[:train_range], f, tensor[train_range:validation_range], output[train_range:validation_range], nEpochs=1000) # Training the NN to be optimized.


train_result = f(tensor[:train_range]) # The training result.
results_s = []
results_b = []

counter = 0
for i in train_result: # This loop is to sort true signal from true background.
    if output[counter] > 0:
        results_s.append(float(i))
    else:
        results_b.append(float(i))

    counter += 1

results_s = np.array(results_s)
results_b = np.array(results_b)

counter += int(0.1*Entries)

###################################################
################## TESTING ########################
###################################################

test_results = f(tensor[validation_range:test_range]).detach().numpy()

test_hist, bins_test, patches_test = plt.hist(test_results, alpha=0)

s = []
b = []
for i in test_results:

    if output[counter] > 0:
        s.append(float(i))
    else:
        b.append(float(i))

    counter += 1

s = np.array(s)
b = np.array(b)



bins = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
centers = bins[1:] - 0.05

plt.figure()
s_hist, s_bins, s_patches = plt.hist(results_s, color="orange", alpha = 0.8, label="True signal")
b_hist, b_bins, b_patches = plt.hist(results_b, color="b", alpha = 0.7, label="True background")

plt.legend()
plt.xlabel("Classified value (Training set)")
plt.ylabel("Counts")
plt.savefig("D:/Users/dunca/Desktop/2023/Modules/DS/Project/Training_classification.png")
print("File succesfully saved.")


plt.figure()
plt.plot(losses, color="b", label="Training set")
plt.plot(losses_validation, color="orange", label="Validation set")
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Binary Cross Entropy")
plt.savefig("D:/Users/dunca/Desktop/2023/Modules/DS/Project/LossFN.png")
print("File succesfully saved.")



s_hist_np = np.histogram(s, bins=bins)
b_hist_np = np.histogram(b, bins=bins)

sumtest = sum(s) + sum(b)
sumtrain = sum(results_s) + sum(results_b)

plt.figure()
sig_hist = plt.hist(s, bins=bins, density=True, stacked=True, color="red", edgecolor="k", alpha=0.5, label="Test signal")
bg_hist = plt.hist(b, bins=bins, density=True, stacked=True, color="blue", edgecolor="k", alpha=0.5, label="Test background")
train_sig_hist = plt.hist(results_s, bins=bins, density=True, stacked=True, color="darkorange", edgecolor="k", alpha=0.5, label="Trained signal")
train_bg_hist = plt.hist(results_b, bins=bins, density=True, stacked=True, color="cyan", edgecolor="k", alpha=0.5, label="Trained background")
plt.errorbar(centers, sig_hist[0], yerr=np.sqrt(s_hist_np[0]/sumtest), ls="", color="r", label="Signal error on test set")
plt.errorbar(centers, bg_hist[0], yerr=np.sqrt(b_hist_np[0]/sumtest), ls="", color="b", label="Background error on test set")


plt.legend()
plt.xlabel("Classification value")
plt.ylabel("Normalized counts")
plt.ylim(0,10)
plt.savefig("D:/Users/dunca/Desktop/2023/Modules/DS/Project/Training_and_test.png")
print("File succesfully saved.")


sig_hist = np.array(sig_hist[0])
bg_hist =  np.array(bg_hist[0])
train_sig_hist = np.array(train_sig_hist[0])
train_bg_hist = np.array(train_bg_hist[0])
print("For each bin in the normalized histograms, we can compute the relative difference of each classification:")
for i in range(len(train_bg_hist)):
    print("Signal difference is: ", np.round(np.abs(train_sig_hist[i]-sig_hist[i]),3), " and the background difference is: ", np.round(np.abs(train_bg_hist[i]-bg_hist[i]),3), "." , sep="")




##########################################################
################## KAGGLE SUBMISSION #####################
##########################################################


df_testset = pd.read_csv("D:/Users/dunca/Desktop/2023/Modules/DS/Project/higgs-boson/test/test.csv")


labels_testset = df_testset.columns[1:] # Dataframe headings.
tensor_testset = torch.tensor(df_testset[labels_testset].values.astype(float)).float() # Here we have our tensor
testset_results = f(tensor_testset).detach().numpy()



IDshift = 350000

final = []
for i in range(len(testset_results)):
    final.append(float(testset_results[i][0]))


final = np.array(final)
index = np.argsort(final)

ranks = np.empty_like(index)
ranks[index] = np.arange(len(final))

# print(final[:20], index[:20])
# print(testset_results_ordered[:-1:])
submission = [], [], []
for i in range(len(final)):
    submission[0].append(i + IDshift)
    submission[1].append(ranks[i]+1)
    if final[i] >= 0.5:
        submission[2].append("s")
    else:
        submission[2].append("b")

data = []
for i in range(len(final)):
    data.append([submission[0][i],submission[1][i],submission[2][i]])

with open("D:/Users/dunca/Desktop/2023/Modules/DS/Project/submission.csv", "w", encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["EventID","RankOrder","Class"])
    writer.writerows(data)

## How much time does file

end = time.time()
execution_time = end - start
execution_time = np.round(execution_time, 0)
seconds = np.mod(execution_time, 60)
minutes = (execution_time - seconds)/60
print("Execution time: ", minutes, " minutes & ", seconds, " seconds." ,sep = "" )

#
