print("Starting imports")
import torch
from dadaptation import DAdaptAdam, DAdaptSGD
import sys
import os
import re
import pydicom
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.functional import pad
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from skimage import filters
import torchvision.transforms as T

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
parent_dir = os.path.dirname(current_path)
sys.path.insert(0, parent_dir)
from training.training_config import *
from models.architectures import *
from models.dataloading import *
from data_processing.data_analysis import *
from data_processing.dcm_processing import *
from data_processing.plotting import *

if on_CSF:
    config = int(sys.argv[1]) - 1
    print(sys.argv[1])
    print(config)
    config = configurations[config]
    batch_size = config['batch_size']
    op_choice = config['optimizer']
    weighted = config['weighted']
    transformed = config['transformed']
    lr = config['lr']

# time.sleep(60*60*14)
print(time.localtime())
seed_value = 272727
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sns.set(style='dark')

# Initialize model, criterion, optimizer
# model = SimpleCNN().to(device)
#edit cuda
model = ResNetTransformer().to(device)
epsilon = 0.
# model = TransformerModel(epsilon=epsilon).to(device)
criterion = nn.MSELoss(reduction='none')  # Mean squared error for regression
if op_choice == 'd_adam':
    optimizer = DAdaptAdam(model.parameters(), lr=lr)
elif op_choice == 'd_sgd':
    optimizer = DAdaptSGD(model.parameters(), lr=lr)
elif op_choice == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr)
else:
    optimizer = optim.SGD(model.parameters(),
                             lr=lr, momentum=momentum)


# Training parameters
not_improved = 0
not_improved_r2 = 0
best_val_loss = float('inf')
best_test_loss = float('inf')
best_val_l_r2 = -float('inf')
best_test_l_r2 = -float('inf')
best_val_r_loss = float('inf')
best_test_r_loss = float('inf')
best_val_r2 = -float('inf')
best_test_r2 = -float('inf')
# scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(patience/10), factor=0.9, verbose=True)
writer = SummaryWriter(working_dir + '/results/' + best_model_name)

print("Beginning training")
for epoch in tqdm(range(num_epochs)):
    model.train()
    all_targets = []
    all_predictions = []
    train_loss = 0.0
    scaled_train_loss = 0.0
    #edit
    for inputs, targets, weights, dir, view in train_loader:  # Simplified unpacking
        inputs, targets, weights = inputs.to(device), targets.to(device), weights.to(device)  # Send data to GPU
        if torch.sum(torch.isnan(inputs)) > 0:
            print("Image is corrupted")

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(inputs.unsqueeze(1))  # Add channel dimension
        losses = criterion(outputs.squeeze(1), targets.float())  # Get losses for each sample
        weighted_loss = (losses * weights).mean()  # Weighted loss

        # Backward + optimize
        weighted_loss.backward()
        optimizer.step()

        train_loss += weighted_loss.item() * inputs.size(0)

        with torch.no_grad():
            train_outputs_original_scale = inverse_standardize_targets(outputs.squeeze(1), mean, std)
            train_targets_original_scale = inverse_standardize_targets(targets.float(), mean, std)
            all_targets.extend(train_targets_original_scale.cpu().numpy())
            all_predictions.extend(train_outputs_original_scale.cpu().numpy())
            scaled_train_loss += criterion(train_outputs_original_scale,
                                           train_targets_original_scale).mean().item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)
    scaled_train_loss /= len(train_loader.dataset)

    train_r2 = r2_score(all_targets, all_predictions)
    # Validation
    val_loss, val_labels, val_preds, val_r2 = evaluate_model(model, val_loader, criterion,
                                                             inverse_standardize_targets, mean, std)
    test_loss, test_labels, test_preds, test_r2 = evaluate_model(model, test_loader, criterion,
                                                                 inverse_standardize_targets, mean, std)
    print(f"Epoch {epoch + 1}/{num_epochs}, "
          f"\nTrain Loss: {scaled_train_loss:.4f}, Val Loss: {val_loss:.4f}, Test loss: {test_loss:.4f}"
          f"\nTrain R2: {train_r2:.4f}, Val R2: {val_r2:.4f}, Test R2: {test_r2:.4f}")

    writer.add_scalar('Loss/Train', scaled_train_loss, epoch)
    writer.add_scalar('R2/Train', train_r2, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('R2/Validation', val_r2, epoch)
    writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('R2/Test', test_r2, epoch)

    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        best_test_r2 = test_r2
        best_val_r_loss = val_loss
        best_test_r_loss = test_loss
        not_improved_r2 = 0
        print("Validation R2 improved. Saving best_model.")
        torch.save(model.state_dict(), working_dir + '/models/r_' + best_model_name)
    else:
        not_improved_r2 += 1
    # Check early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_test_loss = test_loss
        best_val_l_r2 = val_r2
        best_test_l_r2 = test_r2
        not_improved = 0
        print("Validation loss improved. Saving best_model.")
        print(f"From best val loss at epoch {epoch - not_improved}:\n "
              f"val loss: {best_val_loss:.4f} test loss {best_test_loss:.4f} val r2: {best_val_l_r2:.4f} test r2 {best_test_l_r2:.4f}")
        print(f"From best val R2 at epoch {epoch - not_improved_r2}:\n "
              f"val loss: {best_val_r_loss:.4f} test loss {best_test_r_loss:.4f} val r2: {best_val_r2:.4f} test r2 {best_test_r2:.4f}")
        torch.save(model.state_dict(), working_dir + '/models/l_' + best_model_name)
    else:
        not_improved += 1
        print(f"From best val loss at epoch {epoch - not_improved}:\n "
              f"val loss: {best_val_loss:.4f} test loss {best_test_loss:.4f} val r2: {best_val_l_r2:.4f} test r2 {best_test_l_r2:.4f}")
        print(f"From best val R2 at epoch {epoch - not_improved_r2}:\n "
              f"val loss: {best_val_r_loss:.4f} test loss {best_test_r_loss:.4f} val r2: {best_val_r2:.4f} test r2 {best_test_r2:.4f}")
        if not_improved >= patience:
            print("Early stopping")
            break

    writer.add_scalar('Loss/Best Validation Loss from Loss', best_val_loss, epoch)
    writer.add_scalar('Loss/Best Validation Loss from R2', best_val_r_loss, epoch)
    writer.add_scalar('R2/Best Validation R2 from R2', best_val_r2, epoch)
    writer.add_scalar('R2/Best Validation R2 from Loss', best_val_l_r2, epoch)
    writer.add_scalar('Loss/Best Test Loss from Loss', best_test_loss, epoch)
    writer.add_scalar('Loss/Best Test Loss from R2', best_test_r_loss, epoch)
    writer.add_scalar('R2/Best Test R2 from R2', best_test_r2, epoch)
    writer.add_scalar('R2/Best Test R2 from Loss', best_test_l_r2, epoch)

    # scheduler.step(val_loss)

writer.close()
print("Loading best model weights!")
model.load_state_dict(torch.load(working_dir + '/models/l_' + best_model_name))

train_dataset.dataset = MammogramDataset(processed_dataset_path, transform=None)

# Evaluating on all datasets: train, val, test
train_loss, train_labels, train_preds, train_r2 = evaluate_model(model, train_loader, criterion,
                                                                 inverse_standardize_targets, mean, std)
val_loss, val_labels, val_preds, val_r2 = evaluate_model(model, val_loader, criterion, inverse_standardize_targets,
                                                         mean, std)
test_loss, test_labels, test_preds, test_r2 = evaluate_model(model, test_loader, criterion, inverse_standardize_targets,
                                                             mean, std)

# R2 Scores
print(f"Train R2 Score: {train_r2:.4f}")
print(f"Train Loss: {train_loss:.4f}")
print(f"Validation R2 Score: {val_r2:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Test R2 Score: {test_r2:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Scatter plots
plot_scatter(train_labels, train_preds, "Train Scatter Plot " + best_model_name, working_dir + '/results/')
plot_scatter(val_labels, val_preds, "Validation Scatter Plot " + best_model_name, working_dir + '/results/')
plot_scatter(test_labels, test_preds, "Test Scatter Plot " + best_model_name, working_dir + '/results/')

# Error distributions
plot_error_vs_vas(train_labels, train_preds, "Train Error vs VAS " + best_model_name, working_dir + '/results/')
plot_error_vs_vas(val_labels, val_preds, "Validation Error vs VAS " + best_model_name, working_dir + '/results/')
plot_error_vs_vas(test_labels, test_preds, "Test Error vs VAS " + best_model_name, working_dir + '/results/')

# Error distributions
plot_error_distribution(train_labels, train_preds, "Train Error Distribution " + best_model_name,
                        working_dir + '/results/')
plot_error_distribution(val_labels, val_preds, "Validation Error Distribution " + best_model_name,
                        working_dir + '/results/')
plot_error_distribution(test_labels, test_preds, "Test Error Distribution " + best_model_name,
                        working_dir + '/results/')

print("Done")
