import os
import numpy as np
from tqdm import tqdm
import random
import cv2
import albumentations as A
from IPython.display import clear_output
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

def create_dataset(path, monkey_classes):
    x = []
    y = []
    for i in tqdm(range(len(monkey_classes))):
        path_class = path + '/n' + str(i)
        list_class = os.listdir(path_class)
        for item in list_class:
            path_item = path_class + '/' + item
            image = cv2.imread(path_item)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented_image = transform_resize(image=image)['image']
            x.append((torch.from_numpy(augmented_image).float()/255).permute(2, 0, 1))
            y.append(i)
    return x, y

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    
def augment_batch(x):
    b_s = x.shape[0]
    x_aug = torch.zeros(x.shape)
    for i in range(b_s):
        x_aug[i] = torch.from_numpy(transform(image=np.array(x[i].permute(1, 2, 0)))['image']).permute(2, 0, 1)
    return x_aug

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, device, dataloader, optimizer, criterion, clip, train_history=None, valid_history=None):
    model.train()
    
    epoch_loss = 0
    history = []
    iter = 0

    for x, y in dataloader:
        x = augment_batch(x)
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x)

        loss = criterion(output, y)
        loss.backward()
        
        # Let's clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        history.append(loss.cpu().data.numpy())

        if (iter+1)%10==0:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

            clear_output(True)
            ax[0].plot(history, label='train loss')
            ax[0].set_xlabel('Batch')
            ax[0].set_title('Train loss')
            if train_history is not None:
                ax[1].plot(train_history, label='general train history')
                ax[1].set_xlabel('Epoch')
            if valid_history is not None:
                ax[1].plot(valid_history, label='general valid history')
            plt.legend()
            
            plt.show()

        iter += 1

    return epoch_loss / len(dataloader)

def evaluate(model, device, dataloader, criterion):
    
    model.eval()
    epoch_loss = 0
    history = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            
            epoch_loss += loss.item()
            history.append(loss.cpu().data.numpy())
        
    return epoch_loss / len(dataloader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def test_accuracy(model, device, dataloader, weights):
    model.load_state_dict(torch.load(weights))
    model.eval()
    accuracy = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            output = torch.argmax(output, dim=1)
            accuracy += torch.count_nonzero(output==y).item()
        
    return accuracy*100 / len(dataloader) / y.shape[0]
