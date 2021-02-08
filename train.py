import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import torch.optim as optim
from torchvision import datasets, transforms

from network import CNN
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def train():
    ############    Set all params    ############
    lr = 5e-5
    epochs = 7
    batch_size = 100
    img_size = 64 # TODO : Figure out what this really should be
    train_size = 0.9
    weight_decay = 0.0001

    # check if gpu is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using ", device)

    # Load all data in
    dataset = datasets.ImageFolder(root='./data',
                           transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    print("Auto created classes: ", dataset.class_to_idx)
                           
    # split data
    n = len(dataset)
    print(f"Loaded {n} datapoints")
    n_train = int(train_size * n)
    data_train, data_val = random_split(dataset, [n_train, n-n_train])

    # Make loaders
    train_loader = DataLoader(data_train, batch_size=batch_size, pin_memory=True)
    val_loader = DataLoader(data_val, batch_size=batch_size)
    
    # init model
    c, h, w = 3, img_size, img_size
    model = CNN(c,h,w).to(device)

    # init optimizer and other things
    objective = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # set up places to save things
    train_accur = []
    temp_train_accur = []
    val_accur = []
    num_batch = len(train_loader)
    loop = tqdm(total=num_batch*epochs, position=0)

    # do the training!
    for epoch in range(epochs):
        #iterate through batches
        for batch, (x, y_truth) in enumerate(train_loader):
            x, y_truth = x.to(device), y_truth.to(device)
            
            opt.zero_grad()
            y_hat = model(x)

            loss = objective(y_hat, y_truth)
            loss.backward()
            opt.step()
            
            temp_train_accur.append((y_hat.argmax(1) == y_truth).float().mean().item())

            loop.set_description(f"accur: {temp_train_accur[-1]}, loss: {loss.item()}")
            loop.update(1)
                    
            #check validation loss twice per epoch
            if batch % (num_batch//2) == 0 and batch != 0:
                train_accur.append( (epoch+batch/num_batch, np.mean(temp_train_accur)) )
                temp_train_accur = []
                
                temp = [ (model(x.to(device)).argmax(1) == y.to(device)).float().mean().item() for x, y in val_loader]
                val_accur.append( (epoch+batch/num_batch, np.mean(temp)) )

    # plot results
    plt.figure(figsize=(10,6))
    l1, l2 = zip(*train_accur)
    plt.plot(l1, l2, label="Training Accuracy")
    l1, l2 = zip(*val_accur)
    plt.plot(l1, l2, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # save model
    torch.save(model.state_dict(), 'model.pkl')

if __name__ == "__main__":
    train()