from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluation import evaluate_model


def train_model(
    model,
    train_loader,
    val_loader, 
    test_loader,
    num_epochs,
    optimizer,
    device, 
    save_path=f"./ckpt/model.pt"
):
    """
    Feel free to change the arguments of this function - if necessary.

    Trains the model on the given dataset. Selects the best model based on the
    validation set and saves it to the given path. 
    Inputs: 
        model: The model to train [nn.Module]
        train_loader: The training data loader [DataLoader]
        val_loader: The validation data loader [DataLoader]
        num_epochs: The number of epochs to train for [int]
        optimizer: The optimizer [Any]
        best_of: The metric to use for validation [str: "loss" or "accuracy"]
        device: The device to train on [str: cpu, cuda, or mps]
        save_path: The path to save the model to [str]
    Output:
        Dictionary containing the training and validation losses and accuracies
        at each epoch. Also contains the epoch number of the best model.
    """

    #
    # You can put your training loop here
    #
    best_val_loss = float("-inf")
    patience_counter = 0
    patience_limit = 2

    num_features = model.linear.in_features
    model.linear = nn.Linear(num_features, 10) 

    for param in model.parameters():
      param.requires_grad = False

    for param in model.linear.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
      model.train()  
      for images, labels in train_loader:
          images, labels = images.to(device), labels.to(device)

          outputs = model(images)
          loss = criterion(outputs, labels)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      current_val_loss = evaluate_model(model, val_loader, device)
      print("current_val_loss", current_val_loss)
      if current_val_loss < best_val_loss:
        patience_counter += 1
      else:
        best_val_loss = current_val_loss
        patience_counter = 0

      if (patience_counter >= patience_limit):
        print("Early stopping")
        patience_counter = 0
        break

      print("epoch 1", epoch)

    for layer in [model.layer3, model.layer4]:
      for param in layer.parameters():
        param.requires_grad = True

    additional_epochs = 15
    for epoch in range(additional_epochs):
      model.train()
      for images, labels in train_loader:
          images, labels = images.to(device), labels.to(device)

          outputs = model(images)
          loss = criterion(outputs, labels)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      current_val_loss = evaluate_model(model, val_loader, device)
      print("current_val_loss", current_val_loss)
      if current_val_loss < best_val_loss:
        patience_counter += 1
      else:
        best_val_loss = current_val_loss
        patience_counter = 0

      if patience_counter >= patience_limit  or (current_val_loss > 0.73):
        print("Early stopping")
        break

      print("epoch 2", epoch)
    

    return model




    #unfreezing
    #for layer in [model.layer3, model.layer4]:
      #for param in layer.parameters():
          #param.requires_grad = True
    '''
    for g in optimizer.param_groups:
      g['lr'] = 0.0001'''
