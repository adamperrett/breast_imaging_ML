print("Starting imports")
from dadaptation import DAdaptAdam, DAdaptSGD
import sys
import time
from torch.utils.tensorboard import SummaryWriter
import optuna
import sqlite3
import os
from math import log10, floor

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
parent_dir = os.path.dirname(current_path)
sys.path.insert(0, parent_dir)
from training.training_config import *
from models.architectures import *
from models.dataloading import *
from data_processing.data_analysis import *
# from data_processing.dcm_processing import *
from data_processing.plotting import *

# time.sleep(60*60*14)
print(time.localtime())
seed_value = 27
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sns.set(style='dark')

def round_to_(x, sig_fig=2):
   return round(x, -int(floor(log10(abs(x))))+sig_fig)

def regression_training(trial):
    global base_name
    lr = 3e-5 # trial.suggest_float('lr', 3e-6, 1e-3, log=True)
    op_choice = 'adam' #trial.suggest_categorical('optimiser', ['adam', 'rms', 'sgd'])#, 'd_adam', 'd_sgd'])
    batch_size = 14 #trial.suggest_int('batch_size', 2, 50)
    dropout = 0.3 #trial.suggest_float('dropout', 0, 0.7)
    # arch = trial.suggest_categorical('architecture', ['pvas', 'resnetrans'])
    resnet_size = 34 #trial.suggest_categorical('resent_size', [18, 34, 50])
    pooling_type = 'mean' #trial.suggest_categorical('pooling_type', ['mean', 'max', 'attention'])
    pre_trained = 1 #trial.suggest_categorical('pre_trained', [0, 1])
    replicate = 0 #trial.suggest_categorical('replicate', [0, 1])
    transformed = 0 #trial.suggest_categorical('transformed', [0, 1])
    weight_samples = 0 #trial.suggest_categorical('weight_samples', [0, 1])
    weight_loss = 0 #trial.suggest_categorical('weight_loss', [0, 1])
    data_path = processed_dataset_file
    best_model_name = '{}_lr{}x{}_{}{}_p{}r{}_d{}_{}_t{}_wl{}_ws{}'.format(
        base_name, round_to_(lr), batch_size, resnet_size, pooling_type, pre_trained,
        replicate, round_to_(dropout), op_choice, transformed, weight_loss, weight_samples)

    print("Accessing data from", data_path, "\nConfig", best_model_name)
    print(time.localtime())
    print("Current GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
    train_loader, val_loader, test_loader, seed_value, current_fold = \
        return_crossval_loaders(data_path, transformed,
                                weight_loss, weight_samples, batch_size)
    best_model_name = 'cv{}_{}_'.format(seed_value, current_fold) + best_model_name
    pilot_loader = return_medici_loaders(processed_pilot_file, transformed,
                                          weight_loss, weight_samples, batch_size,
                                          only_testing=True)

    num_manufacturers = 8
    manufacturer_mapping = {'Philips': nn.functional.one_hot(torch.tensor(0),
                                                             num_classes=num_manufacturers).to(torch.float32).to('cuda'),
                            'SIEMENS': nn.functional.one_hot(torch.tensor(1),
                                                             num_classes=num_manufacturers).to(torch.float32).to('cuda'),
                            'HOLOGIC': nn.functional.one_hot(torch.tensor(2),
                                                             num_classes=num_manufacturers).to(torch.float32).to('cuda'),
                            'GE': nn.functional.one_hot(torch.tensor(3),
                                                             num_classes=num_manufacturers).to(torch.float32).to('cuda'),
                            'KODAK': nn.functional.one_hot(torch.tensor(4),
                                                             num_classes=num_manufacturers).to(torch.float32).to('cuda'),
                            'FUJIFILM': nn.functional.one_hot(torch.tensor(5),
                                                             num_classes=num_manufacturers).to(torch.float32).to('cuda'),
                            'IMS': nn.functional.one_hot(torch.tensor(6),
                                                             num_classes=num_manufacturers).to(torch.float32).to('cuda'),
                            'LORAD': nn.functional.one_hot(torch.tensor(7),
                                                             num_classes=num_manufacturers).to(torch.float32).to('cuda')
                            }

    # Initialize model, criterion, optimizer
    # model = SimpleCNN().to(device)
    #edit cuda
    print("Loading models\nCurrent GPU mem usage is", torch.cuda.memory_allocated() / (1024 ** 2))
    model = Medici_MIL_Model(pre_trained, replicate, resnet_size, pooling_type, dropout,
                             split=split_CC_and_MLO, num_manufacturers=num_manufacturers).to('cuda')
    epsilon = 0.
    # model = TransformerModel(epsilon=epsilon).to(device)
    criterion = nn.MSELoss(reduction='none')  # Mean squared error for regression
    if op_choice == 'd_adam':
        optimizer = DAdaptAdam(model.parameters())
    elif op_choice == 'd_sgd':
        optimizer = DAdaptSGD(model.parameters())
    elif op_choice == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif op_choice == 'rms':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(),
                                 lr=lr, momentum=momentum)


    # Training parameters
    not_improved_loss = 0
    not_improved_r2 = 0
    not_improved_r2w = 0
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    best_pilot_loss = float('inf')
    best_val_l_r2 = -float('inf')
    best_test_l_r2 = -float('inf')
    best_pilot_l_r2 = -float('inf')
    best_val_rw_r2 = -float('inf')
    best_test_rw_r2 = -float('inf')
    best_pilot_rw_r2 = -float('inf')
    best_val_l_r2w = -float('inf')
    best_test_l_r2w = -float('inf')
    best_pilot_l_r2w = -float('inf')
    best_val_r_r2w = -float('inf')
    best_test_r_r2w = -float('inf')
    best_pilot_r_r2w = -float('inf')
    best_val_r_loss = float('inf')
    best_test_r_loss = float('inf')
    best_pilot_r_loss = float('inf')
    best_val_rw_loss = float('inf')
    best_test_rw_loss = float('inf')
    best_pilot_rw_loss = float('inf')
    best_val_r2 = -float('inf')
    best_test_r2 = -float('inf')
    best_pilot_r2 = -float('inf')
    best_val_r2w = -float('inf')
    best_test_r2w = -float('inf')
    best_pilot_r2w = -float('inf')
    best_train_error_from_loss = 0
    best_train_conf_int_from_loss = 0
    best_train_error_from_r2 = 0
    best_train_conf_int_from_r2 = 0
    best_train_error_from_r2w = 0
    best_train_conf_int_from_r2w = 0
    best_val_error_from_loss = 0
    best_val_conf_int_from_loss = 0
    best_val_error_from_r2 = 0
    best_val_conf_int_from_r2 = 0
    best_val_error_from_r2w = 0
    best_val_conf_int_from_r2w = 0
    best_test_error_from_loss = 0
    best_test_conf_int_from_loss = 0
    best_test_error_from_r2 = 0
    best_test_conf_int_from_r2 = 0
    best_test_error_from_r2w = 0
    best_test_conf_int_from_r2w = 0
    best_pilot_error_from_loss = 0
    best_pilot_conf_int_from_loss = 0
    best_pilot_error_from_r2 = 0
    best_pilot_conf_int_from_r2 = 0
    best_pilot_error_from_r2w = 0
    best_pilot_conf_int_from_r2w = 0
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(patience/10), factor=0.9, verbose=True)
    writer = SummaryWriter(working_dir + '/results/' + best_model_name)

    print("Beginning training", time.localtime())
    print("Current GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
    for epoch in tqdm(range(num_epochs)):
        model.train()
        all_targets = []
        all_predictions = []
        train_loss = 0.0
        scaled_train_loss = 0.0
        for inputs, targets, weight, patient, manu, views in tqdm(train_loader):  # Simplified unpacking
            inputs, targets, weights = inputs.to('cuda'), targets.to('cuda'), targets.to('cuda')  # Send data to GPU
            # print("inputs, targets, weights, dir, view")
            # print(inputs, "\n", targets, "\n", weights, "\n", dir, "\n", view)
            print(f"Loaded images{best_model_name}\nCurrent GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
            if torch.sum(torch.isnan(inputs)) > 0:
                print("Image is corrupted", torch.sum(torch.isnan(inputs), dim=1))
                nan_mask = torch.isnan(inputs)
                inputs[nan_mask] = 0

            # Zero the parameter gradients
            optimizer.zero_grad()

            is_it_mlo = torch.zeros([len(views[0]), len(views), 2]).float().to('cuda')
            if not split_CC_and_MLO:
                for i in range(len(views[0])):
                    for j in range(len(views)):
                        if 'MLO' in views[j][i] or 'mlo' in views[j][i]:
                            is_it_mlo[i][j][0] += 1
                        else:
                            is_it_mlo[i][j][1] += 1
            manufacturer = torch.zeros([len(views[0]), len(views), num_manufacturers]).float().to('cuda')
            for j in range(len(manufacturer[0])):
                for i in range(len(manufacturer)):
                    manufacturer[i][j] += manufacturer_mapping[manu[i]]

            # Forward
            print("Before output\nCurrent GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
            outputs = model.forward(inputs.unsqueeze(1), is_it_mlo, manufacturer)  # Add channel dimension
            print("Before losses\nCurrent GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
            losses = criterion(outputs.squeeze(1), targets.float())  # Get losses for each sample
            print("Before weighting\nCurrent GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
            weighted_loss = (losses * weights).mean()  # Weighted loss

            # Backward + optimize
            print("Before backward\nCurrent GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
            weighted_loss.backward()
            optimizer.step()

            train_loss += weighted_loss.item() * inputs.size(0)

            print("Before scaling loss\nCurrent GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
            with torch.no_grad():
                train_outputs_original_scale = outputs.squeeze(1) #inverse_standardize_targets(outputs.squeeze(1), mean, std)
                train_targets_original_scale = targets.float() #inverse_standardize_targets(targets.float(), mean, std)
                all_targets.extend(train_targets_original_scale.cpu().numpy())
                all_predictions.extend(train_outputs_original_scale.cpu().numpy())
                scaled_train_loss += criterion(train_outputs_original_scale,
                                               train_targets_original_scale).mean().item() * inputs.size(0)
            # break
        train_loss /= len(train_loader.dataset)
        scaled_train_loss /= len(train_loader.dataset)

        train_r2 = r2_score(all_targets, all_predictions)
        train_r2w = r2_score(all_targets, all_predictions, sample_weight=np.array(all_targets)+r2_weighting_offset)
        train_err, train_stderr, train_conf = compute_error_metrics(all_targets, all_predictions)
        # Validation
        print("Evaluating on the validation set")
        val_loss, val_labels, val_preds, val_r2, val_r2w, val_err, val_conf, val_manu = \
            evaluate_medici(model, val_loader, criterion,
                            inverse_standardize_targets, mean, std,
                            num_manufacturers=num_manufacturers,
                            manufacturer_mapping=manufacturer_mapping,
                            split_CC_and_MLO=split_CC_and_MLO,
                            r2_weighting_offset=r2_weighting_offset
                            )
        print("Evaluating on the test set")
        test_loss, test_labels, test_preds, test_r2, test_r2w, test_err, test_conf, test_manu = \
            evaluate_medici(model, test_loader, criterion,
                            inverse_standardize_targets, mean, std,
                            num_manufacturers=num_manufacturers,
                            manufacturer_mapping=manufacturer_mapping,
                            split_CC_and_MLO=split_CC_and_MLO,
                            r2_weighting_offset=r2_weighting_offset
                            )
        print("Evaluating on the pilot set")
        pilot_loss, pilot_labels, pilot_preds, pilot_r2, pilot_r2w, pilot_err, pilot_conf, pilot_manu = \
            evaluate_medici(model, pilot_loader, criterion,
                            inverse_standardize_targets, mean, std,
                            num_manufacturers=num_manufacturers,
                            manufacturer_mapping=manufacturer_mapping,
                            split_CC_and_MLO=split_CC_and_MLO,
                            r2_weighting_offset=r2_weighting_offset)
        val_fig = plot_scatter(val_labels, val_preds, "Validation Scatter Plot " + best_model_name, False, True,
                               manufacturers=val_manu)
        writer.add_figure("Validation Scatter Plot/{}".format(best_model_name), val_fig, epoch)
        test_fig = plot_scatter(test_labels, test_preds, "Test Scatter Plot " + best_model_name, False, True,
                                manufacturers=test_manu)
        writer.add_figure("Test Scatter Plot/{}".format(best_model_name), test_fig, epoch)
        pilot_fig = plot_scatter(pilot_labels, pilot_preds, "Pilot Scatter Plot " + best_model_name, False, True,
                                 manufacturers=pilot_manu)
        writer.add_figure("Pilot Scatter Plot/{}".format(best_model_name), pilot_fig, epoch)
        print(f"Epoch {epoch + 1}/{num_epochs}, {best_model_name}"
              f"\nTrain Loss: {scaled_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Test loss: {test_loss:.4f}, Pilot loss: {pilot_loss:.4f}"
              f"\nTrain R2: {train_r2:.4f}, Val R2: {val_r2:.4f}, "
              f"Test R2: {test_r2:.4f}, Pilot R2: {pilot_r2:.4f}"
              f"\nTrain R2w: {train_r2w:.4f}, Val R2w: {val_r2w:.4f}, "
              f"Test R2w: {test_r2w:.4f}, Pilot R2w: {pilot_r2w:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}, {best_model_name}"
              f"\nTrain Loss: {scaled_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Test loss: {test_loss:.4f}"
              f"\nTrain R2: {train_r2:.4f}, Val R2: {val_r2:.4f}, "
              f"Test R2: {test_r2:.4f}"
              f"\nTrain R2w: {train_r2w:.4f}, Val R2w: {val_r2w:.4f}, "
              f"Test R2w: {test_r2w:.4f}")

        writer.add_scalar('Loss/Train', scaled_train_loss, epoch)
        writer.add_scalar('R2/Train', train_r2, epoch)
        writer.add_scalar('R2 weighted/Train', train_r2w, epoch)
        writer.add_scalar('R2/Train', train_r2, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('R2/Validation', val_r2, epoch)
        writer.add_scalar('R2 weighted/Validation', val_r2w, epoch)
        writer.add_scalar('Error/Mean error', train_err, epoch)
        writer.add_scalar('Error/Confidence interval', train_conf, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('R2/Test', test_r2, epoch)
        writer.add_scalar('R2 weighted/Test', test_r2w, epoch)
        writer.add_scalar('Error/Mean error', train_err, epoch)
        writer.add_scalar('Error/Confidence interval', train_conf, epoch)
        writer.add_scalar('Loss/Pilot', pilot_loss, epoch)
        writer.add_scalar('R2/Pilot', pilot_r2, epoch)
        writer.add_scalar('R2 weighted/Pilot', pilot_r2w, epoch)

        writer.add_scalar('Error/Mean error training', train_err, epoch)
        writer.add_scalar('Error/Confidence interval training', train_conf, epoch)
        writer.add_scalar('Error/Mean error validation', val_err, epoch)
        writer.add_scalar('Error/Confidence interval validation', val_conf, epoch)
        writer.add_scalar('Error/Mean error testing', test_err, epoch)
        writer.add_scalar('Error/Confidence interval testing', test_conf, epoch)
        writer.add_scalar('Error/Mean error pilot', pilot_err, epoch)
        writer.add_scalar('Error/Confidence interval pilot', pilot_conf, epoch)

        if on_CSF and optuna_optimisation:
            for attempt in range(40):
                try:
                    if improving_loss_or_r2 == 'r2':
                        trial.report(val_r2, epoch)
                    else:
                        trial.report(val_loss, epoch)
                    if trial.should_prune():
                        print("Pruning", best_model_name)
                        raise optuna.TrialPruned()
                except (sqlite3.OperationalError, optuna.exceptions.StorageInternalError) as e:
                    print(f"Attempt {attempt + 1} failed with database lock error: {e}. Retrying in {20} seconds...")
                    time.sleep(20)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_test_r2 = test_r2
            best_pilot_r2 = pilot_r2
            best_val_r_loss = val_loss
            best_test_r_loss = test_loss
            best_pilot_r_loss = pilot_loss
            best_val_r_r2w = val_r2w
            best_test_r_r2w = test_r2w
            best_pilot_r_r2w = pilot_r2w
            best_train_error_from_r2 = train_err
            best_train_conf_int_from_r2 = train_conf
            best_val_error_from_r2 = val_err
            best_val_conf_int_from_r2 = val_conf
            best_test_error_from_r2 = test_err
            best_test_conf_int_from_r2 = test_conf
            best_pilot_error_from_r2 = pilot_err
            best_pilot_conf_int_from_r2 = pilot_conf
            not_improved_r2 = 0
            print("Validation R2 improved. Saving best_model.")
            torch.save(model.state_dict(), working_dir + '/../models/rw_' + best_model_name)
        else:
            not_improved_r2 += 1
        if val_r2w > best_val_r2w:
            best_val_r2w = val_r2w
            best_test_r2w = test_r2w
            best_pilot_r2w = pilot_r2w
            best_val_rw_loss = val_loss
            best_test_rw_loss = test_loss
            best_pilot_rw_loss = pilot_loss
            best_val_rw_r2 = val_r2
            best_test_rw_r2 = test_r2
            best_pilot_rw_r2 = pilot_r2
            best_train_error_from_r2w = train_err
            best_train_conf_int_from_r2w = train_conf
            best_val_error_from_r2w = val_err
            best_val_conf_int_from_r2w = val_conf
            best_test_error_from_r2w = test_err
            best_test_conf_int_from_r2w = test_conf
            best_pilot_error_from_r2w = pilot_err
            best_pilot_conf_int_from_r2w = pilot_conf
            not_improved_r2w = 0
            print("Validation R2 weighted improved. Saving best_model.")
            torch.save(model.state_dict(), working_dir + '/../models/r_' + best_model_name)
        else:
            not_improved_r2w += 1
        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            best_pilot_loss = pilot_loss
            best_val_l_r2 = val_r2
            best_test_l_r2 = test_r2
            best_pilot_l_r2 = pilot_r2
            best_val_l_r2w = val_r2w
            best_test_l_r2w = test_r2w
            best_pilot_l_r2w = pilot_r2w
            best_train_error_from_loss = train_err
            best_train_conf_int_from_loss = train_conf
            best_val_error_from_loss = val_err
            best_val_conf_int_from_loss = val_conf
            best_test_error_from_loss = test_err
            best_test_conf_int_from_loss = test_conf
            best_pilot_error_from_loss = pilot_err
            best_pilot_conf_int_from_loss = pilot_conf
            not_improved_loss = 0
            print("Validation loss improved. Saving best_model.")
            torch.save(model.state_dict(), working_dir + '/../models/l_' + best_model_name)
        else:
            not_improved_loss += 1
        print(f"From best val loss at epoch {epoch - not_improved_loss}:\n "
              f"val loss: {best_val_loss:.4f} test loss {best_test_loss:.4f} pilot loss {best_pilot_loss:.4f} "
              f"val r2: {best_val_l_r2:.4f} test r2 {best_test_l_r2:.4f} pilot r2 {best_pilot_l_r2:.4f}"
              f"val r2w: {best_val_l_r2w:.4f} test r2w {best_test_l_r2w:.4f} pilot r2w {best_pilot_l_r2w:.4f}")
        print(f"From best val R2 at epoch {epoch - not_improved_r2}:\n "
              f"val loss: {best_val_r_loss:.4f} test loss {best_test_r_loss:.4f} pilot loss {best_pilot_r_loss:.4f} "
              f"val r2: {best_val_r2:.4f} test r2 {best_test_r2:.4f} pilot r2 {best_pilot_r2:.4f}"
              f"val r2w: {best_val_r_r2w:.4f} test r2w {best_test_r_r2w:.4f} pilot r2w {best_pilot_r_r2w:.4f}")
        print(f"From best val R2w at epoch {epoch - not_improved_r2w}:\n "
              f"val loss: {best_val_rw_loss:.4f} test loss {best_test_rw_loss:.4f} pilot loss {best_pilot_rw_loss:.4f} "
              f"val r2: {best_val_rw_r2:.4f} test r2 {best_test_rw_r2:.4f} pilot r2 {best_pilot_rw_r2:.4f}"
              f"val r2w: {best_val_r2w:.4f} test r2w {best_test_r2w:.4f} pilot r2w {best_pilot_r2w:.4f}")
        if improving_loss_or_r2 == 'r2':
            time_since_improved = not_improved_r2
        elif improving_loss_or_r2 == 'r2w':
            time_since_improved = not_improved_r2w
        else:
            time_since_improved = not_improved_loss

        writer.add_scalar('Loss/Best Validation Loss from Loss', best_val_loss, epoch)
        writer.add_scalar('Loss/Best Validation Loss from R2', best_val_r_loss, epoch)
        writer.add_scalar('Loss/Best Validation Loss from R2w', best_val_rw_loss, epoch)
        writer.add_scalar('R2/Best Validation R2 from R2', best_val_r2, epoch)
        writer.add_scalar('R2/Best Validation R2 from Loss', best_val_l_r2, epoch)
        writer.add_scalar('R2/Best Validation R2 from R2w', best_val_rw_r2, epoch)
        writer.add_scalar('R2w/Best Validation R2w from R2', best_val_r_r2w, epoch)
        writer.add_scalar('R2w/Best Validation R2w from Loss', best_val_l_r2w, epoch)
        writer.add_scalar('R2w/Best Validation R2w from R2w', best_val_r2w, epoch)
        writer.add_scalar('Loss/Best Test Loss from Loss', best_test_loss, epoch)
        writer.add_scalar('Loss/Best Test Loss from R2', best_test_r_loss, epoch)
        writer.add_scalar('Loss/Best Test Loss from R2w', best_test_rw_loss, epoch)
        writer.add_scalar('R2/Best Test R2 from R2', best_test_r2, epoch)
        writer.add_scalar('R2/Best Test R2 from Loss', best_test_l_r2, epoch)
        writer.add_scalar('R2/Best Test R2 from R2w', best_test_rw_r2, epoch)
        writer.add_scalar('R2w/Best Test R2w from R2', best_test_r_r2w, epoch)
        writer.add_scalar('R2w/Best Test R2w from Loss', best_test_l_r2w, epoch)
        writer.add_scalar('R2w/Best Test R2w from R2w', best_test_r_r2w, epoch)
        writer.add_scalar('Loss/Best Pilot Loss from Loss', best_pilot_loss, epoch)
        writer.add_scalar('Loss/Best Pilot Loss from R2', best_pilot_r_loss, epoch)
        writer.add_scalar('Loss/Best Pilot Loss from R2w', best_pilot_rw_loss, epoch)
        writer.add_scalar('R2/Best Pilot R2 from R2', best_pilot_r2, epoch)
        writer.add_scalar('R2/Best Pilot R2 from Loss', best_pilot_l_r2, epoch)
        writer.add_scalar('R2/Best Pilot R2 from R2w', best_pilot_rw_r2, epoch)
        writer.add_scalar('R2w/Best Pilot R2w from R2', best_pilot_r_r2w, epoch)
        writer.add_scalar('R2w/Best Pilot R2w from Loss', best_pilot_l_r2w, epoch)
        writer.add_scalar('R2w/Best Pilot R2w from R2w', best_pilot_r2w, epoch)

        writer.add_scalar('Best Error from loss/Mean error training', best_train_error_from_loss, epoch)
        writer.add_scalar('Best Error from loss/Confidence interval training', best_train_conf_int_from_loss, epoch)
        writer.add_scalar('Best Error from loss/Mean error validation', best_val_error_from_loss, epoch)
        writer.add_scalar('Best Error from loss/Confidence interval validation', best_val_conf_int_from_loss, epoch)
        writer.add_scalar('Best Error from loss/Mean error testing', best_test_error_from_loss, epoch)
        writer.add_scalar('Best Error from loss/Confidence interval testing', best_test_conf_int_from_loss, epoch)
        writer.add_scalar('Best Error from loss/Mean error pilot', best_pilot_error_from_loss, epoch)
        writer.add_scalar('Best Error from loss/Confidence interval pilot', best_pilot_conf_int_from_loss, epoch)
        writer.add_scalar('Best Error from r2/Mean error training', best_train_error_from_r2, epoch)
        writer.add_scalar('Best Error from r2/Confidence interval training', best_train_conf_int_from_r2, epoch)
        writer.add_scalar('Best Error from r2/Mean error validation', best_val_error_from_r2, epoch)
        writer.add_scalar('Best Error from r2/Confidence interval validation', best_val_conf_int_from_r2, epoch)
        writer.add_scalar('Best Error from r2/Mean error testing', best_test_error_from_r2, epoch)
        writer.add_scalar('Best Error from r2/Confidence interval testing', best_test_conf_int_from_r2, epoch)
        writer.add_scalar('Best Error from r2/Mean error pilot', best_pilot_error_from_r2, epoch)
        writer.add_scalar('Best Error from r2/Confidence interval pilot', best_pilot_conf_int_from_r2, epoch)
        writer.add_scalar('Best Error from r2w/Mean error training', best_train_error_from_r2w, epoch)
        writer.add_scalar('Best Error from r2w/Confidence interval training', best_train_conf_int_from_r2w, epoch)
        writer.add_scalar('Best Error from r2w/Mean error validation', best_val_error_from_r2w, epoch)
        writer.add_scalar('Best Error from r2w/Confidence interval validation', best_val_conf_int_from_r2w, epoch)
        writer.add_scalar('Best Error from r2w/Mean error testing', best_test_error_from_r2w, epoch)
        writer.add_scalar('Best Error from r2w/Confidence interval testing', best_test_conf_int_from_r2w, epoch)
        writer.add_scalar('Best Error from r2w/Mean error pilot', best_pilot_error_from_r2w, epoch)
        writer.add_scalar('Best Error from r2w/Confidence interval pilot', best_pilot_conf_int_from_r2w, epoch)

        if time_since_improved >= patience:
            print("Early stopping")
            break
        # scheduler.step(val_loss)

    writer.close()
    print("Loading best model weights!")
    model.load_state_dict(torch.load(working_dir + '/../models/l_' + best_model_name))

    train_loader.dataset.transform = None
    if weight_samples:
        train_loader.sampler.weights = torch.ones_like(train_loader.sampler.weights)
        train_loader.sampler.replacement = False

    # Evaluating on all datasets: train, val, test
    print("Final evaluation on the train set")
    train_loss, train_labels, train_preds, train_r2, train_r2w, train_err, train_conf, train_manu = \
        evaluate_medici(model, train_loader, criterion,
                        inverse_standardize_targets, mean, std,
                        num_manufacturers=num_manufacturers,
                        manufacturer_mapping=manufacturer_mapping,
                        split_CC_and_MLO=split_CC_and_MLO,
                        r2_weighting_offset=r2_weighting_offset
                        )
    print("Final evaluation on the validation set")
    val_loss, val_labels, val_preds, val_r2, val_r2w, val_err, val_conf, val_manu = \
        evaluate_medici(model, val_loader, criterion,
                        inverse_standardize_targets,
                        mean, std,
                        num_manufacturers=num_manufacturers,
                        manufacturer_mapping=manufacturer_mapping,
                        split_CC_and_MLO=split_CC_and_MLO,
                        r2_weighting_offset=r2_weighting_offset
                        )
    print("Final evaluation on the test set")
    test_loss, test_labels, test_preds, test_r2, test_r2w, test_err, test_conf, test_manu, \
    patients, timepoints = \
        evaluate_medici(model, test_loader, criterion,
                        inverse_standardize_targets,
                        mean, std,
                        num_manufacturers=num_manufacturers,
                        manufacturer_mapping=manufacturer_mapping,
                        split_CC_and_MLO=split_CC_and_MLO,
                        r2_weighting_offset=r2_weighting_offset,
                        return_names=True
                        )
    file_path = "all_cross_fold_data.csv"
    df = pd.read_csv(os.path.join(working_dir, file_path))
    for patient, timepoint, prediction in zip(patients, timepoints, test_preds):
        unique_condition = (df['case'] == int(patient)) & (df['timepoint'] == int(timepoint))
        last_column_name = 'crossval{}-{}'.format(seed_value, current_fold) #df.columns[-1]
        df.loc[unique_condition, last_column_name] = prediction
    df.to_csv(os.path.join(working_dir, file_path), index=False)

    print("Final evaluation on the pilot set")
    pilot_loss, pilot_labels, pilot_preds, pilot_r2, pilot_r2w, pilot_err, pilot_conf, pilot_manu = \
        evaluate_medici(model, pilot_loader, criterion,
                        inverse_standardize_targets,
                        mean, std,
                        num_manufacturers=num_manufacturers,
                        manufacturer_mapping=manufacturer_mapping,
                        split_CC_and_MLO=split_CC_and_MLO,
                        r2_weighting_offset=r2_weighting_offset)

    # R2 Scores
    print(f"Train R2 Score: {train_r2:.4f}")
    print(f"Train R2w Score: {train_r2w:.4f}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Error: {train_err:.4f}")
    print(f"Train Confidence interval: {train_conf:.4f}")
    print(f"Validation R2 Score: {val_r2:.4f}")
    print(f"Validation R2w Score: {val_r2w:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Error: {val_err:.4f}")
    print(f"Validation Confidence interval: {val_conf:.4f}")
    print(f"Test R2 Score: {test_r2:.4f}")
    print(f"Test R2w Score: {test_r2w:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Error: {test_err:.4f}")
    print(f"Test Confidence interval: {test_conf:.4f}")
    print(f"Pilot R2 Score: {pilot_r2:.4f}")
    print(f"Pilot R2w Score: {pilot_r2w:.4f}")
    print(f"Pilot Loss: {pilot_loss:.4f}")
    print(f"Pilot Error: {pilot_err:.4f}")
    print(f"Pilot Confidence interval: {pilot_conf:.4f}")

    # Scatter plots
    plot_scatter(train_labels, train_preds, "Train Scatter Plot " + best_model_name, working_dir + '/results/',
                 manufacturers=train_manu)
    plot_scatter(val_labels, val_preds, "Validation Scatter Plot " + best_model_name, working_dir + '/results/',
                 manufacturers=val_manu)
    plot_scatter(test_labels, test_preds, "Test Scatter Plot " + best_model_name, working_dir + '/results/',
                 manufacturers=test_manu)
    plot_scatter(pilot_labels, pilot_preds, "Pilot Scatter Plot " + best_model_name, working_dir + '/results/',
                 manufacturers=pilot_manu)

    # Error distributions
    plot_error_vs_vas(train_labels, train_preds, "Train Error vs VAS " + best_model_name, working_dir + '/results/',
                      manufacturers=train_manu)
    plot_error_vs_vas(val_labels, val_preds, "Validation Error vs VAS " + best_model_name, working_dir + '/results/',
                      manufacturers=val_manu)
    plot_error_vs_vas(test_labels, test_preds, "Test Error vs VAS " + best_model_name, working_dir + '/results/',
                      manufacturers=test_manu)
    plot_error_vs_vas(pilot_labels, pilot_preds, "Pilot Error vs VAS " + best_model_name, working_dir + '/results/',
                      manufacturers=pilot_manu)

    # Error distributions
    plot_error_distribution(train_labels, train_preds, "Train Error Distribution " + best_model_name,
                            working_dir + '/results/', manufacturers=train_manu)
    plot_error_distribution(val_labels, val_preds, "Validation Error Distribution " + best_model_name,
                            working_dir + '/results/', manufacturers=val_manu)
    plot_error_distribution(test_labels, test_preds, "Test Error Distribution " + best_model_name,
                            working_dir + '/results/', manufacturers=test_manu)
    plot_error_distribution(pilot_labels, pilot_preds, "Pilot Error Distribution " + best_model_name,
                            working_dir + '/results/', manufacturers=pilot_manu)

    print("Done")

    if improving_loss_or_r2 == 'r2':
        return np.max(val_r2)
    else:
        return np.min(val_loss)





if __name__ == "__main__":
    if on_CSF and optuna_optimisation:
        print("Setting up optuna optimisation", time.localtime())
        study_name = '{}_cross_validation'.format(base_name)  # Unique identifier
        storage_url = 'sqlite:///{}.db'.format(study_name)
        if improving_loss_or_r2 == 'r2':
            direction = 'maximize'
        else:
            direction = 'minimize'
        study = optuna.create_study(study_name=study_name, storage=storage_url, load_if_exists=True,
                                    # sampler=optuna.samplers.TPESampler,
                                    # sampler=optuna.samplers.NSGAIIISampler(population_size=30), # can do multiple objectives
                                    direction=direction, pruner=optuna.pruners.NopPruner())
        print("Beginning optimisation", time.localtime())
        study.optimize(regression_training, n_trials=1)  # Each script execution does 1 trial

        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_slice,
            plot_contour,
            plot_parallel_coordinate,
            plot_edf,
            plot_intermediate_values,
        )

        # Assuming `study` is your Optuna study object
        optimization_history_fig = plot_optimization_history(study)
        param_importances_fig = plot_param_importances(study)

        results_dir = os.path.join(working_dir, 'results')

        print("Saving figures summarising hyperparameter tuning")

        print("Optimization History")
        fig = plot_optimization_history(study)
        fig.write_html(os.path.join(results_dir, "{}_optimization_history.html".format(study_name)))

        print("Parameter Importances")
        fig = plot_param_importances(study)
        fig.write_html(os.path.join(results_dir, "{}_param_importances.html".format(study_name)))

        print("Slice Plot")
        fig = plot_slice(study)
        fig.write_html(os.path.join(results_dir, "{}_slice_plot.html".format(study_name)))

        print("Contour Plot")
        fig = plot_contour(study)
        fig.write_html(os.path.join(results_dir, "{}_contour_plot.html".format(study_name)))

        print("Parallel Coordinate Plot")
        fig = plot_parallel_coordinate(study)
        fig.write_html(os.path.join(results_dir, "{}_parallel_coordinate.html".format(study_name)))

        print("EDF Plot")
        fig = plot_edf(study)
        fig.write_html(os.path.join(results_dir, "{}_edf_plot.html".format(study_name)))

        print("Intermediate Values")
        fig = plot_intermediate_values(study)
        fig.write_html(os.path.join(results_dir, "{}_intermediate_values.html".format(study_name)))

    else:
        print("Running without optuna", time.localtime())
        _ = regression_training(None)
    print("Done")