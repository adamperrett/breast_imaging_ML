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
    if optuna_optimisation:
        lr = trial.suggest_float('lr', 3e-6, 1e-4, log=True)
        op_choice = 'adam' #trial.suggest_categorical('optimiser', ['adam', 'rms', 'sgd'])#, 'd_adam', 'd_sgd'])
        batch_size = trial.suggest_int('batch_size', 10, 27)
        dropout = trial.suggest_float('dropout', 0, 0.7)
        arch = trial.suggest_categorical('architecture', ['pvas', 'resnetrans'])
        pre_trained = 1 #trial.suggest_categorical('pre_trained', [0, 1])
        replicate = 0 #trial.suggest_categorical('replicate', [0, 1])
        transformed = 0 #trial.suggest_categorical('transformed', [0, 1])
        weight_samples = 0 #trial.suggest_categorical('weight_samples', [0, 1])
        weight_loss = 0 #trial.suggest_categorical('weight_loss', [0, 1])
        data_path = processed_dataset_file
    else:
        lr = 1.28e-05
        op_choice = 'adam' #trial.suggest_categorical('optimiser', ['adam', 'rms', 'sgd'])#, 'd_adam', 'd_sgd'])
        batch_size = 10
        dropout = 0.196
        arch = 'resnetrans'
        pre_trained = 1 #trial.suggest_categorical('pre_trained', [0, 1])
        replicate = 0
        transformed = 0 #trial.suggest_categorical('transformed', [0, 1])
        weight_samples = 0
        weight_loss = 0
        config = int(sys.argv[1]) - 1
        outlier_configs = ['procas_pvas_vas_raw_base']
        string_names = ['no_thresholding']
        for threshold in [2, 6, 10, 14, 18, 22, 26, 30]:
            for method in ['remove_patient', 'remove_image', 'replace_with_avg']:
                outlier_configs.append(f'outlier_processed_by_{method}_threshold_{threshold}_pvas_vas_raw_base')
                string_names.append('{}_th{}'.format(method, threshold))
        data_path = outlier_configs[config]
        base_name = base_name + '_{}'.format(string_names[config])

    best_model_name = '{}_lr{}x{}_{}_p{}r{}_d{}_{}_t{}_wl{}_ws{}'.format(
        base_name, round_to_(lr), batch_size, arch, pre_trained, replicate, round_to_(dropout), op_choice, transformed,
        weight_loss, weight_samples)

    print("Accessing data from", data_path, "\nConfig", best_model_name)
    print(time.localtime())
    print("Current GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
    train_loader, val_loader, test_loader = return_dataloaders(data_path, transformed,
                                                               weight_loss, weight_samples, batch_size)
    priors_loader = return_dataloaders(processed_priors_file, transformed,
                                       weight_loss, weight_samples, batch_size,
                                       only_testing=True)

    # Initialize model, criterion, optimizer
    # model = SimpleCNN().to(device)
    #edit cuda
    print("Loading models\nCurrent GPU mem usage is", torch.cuda.memory_allocated() / (1024 ** 2))
    if arch == 'pvas':
        model = Pvas_Model(pre_trained, replicate, dropout, split=split_CC_and_MLO).to('cuda')
    else:
        model = ResNetTransformer(pre_trained, replicate, dropout, split=split_CC_and_MLO).to('cuda')
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
    best_priors_loss = float('inf')
    best_val_l_r2 = -float('inf')
    best_test_l_r2 = -float('inf')
    best_priors_l_r2 = -float('inf')
    best_val_rw_r2 = -float('inf')
    best_test_rw_r2 = -float('inf')
    best_priors_rw_r2 = -float('inf')
    best_val_l_r2w = -float('inf')
    best_test_l_r2w = -float('inf')
    best_priors_l_r2w = -float('inf')
    best_val_r_r2w = -float('inf')
    best_test_r_r2w = -float('inf')
    best_priors_r_r2w = -float('inf')
    best_val_r_loss = float('inf')
    best_test_r_loss = float('inf')
    best_priors_r_loss = float('inf')
    best_val_rw_loss = float('inf')
    best_test_rw_loss = float('inf')
    best_priors_rw_loss = float('inf')
    best_val_r2 = -float('inf')
    best_test_r2 = -float('inf')
    best_priors_r2 = -float('inf')
    best_val_r2w = -float('inf')
    best_test_r2w = -float('inf')
    best_priors_r2w = -float('inf')
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
    best_priors_error_from_loss = 0
    best_priors_conf_int_from_loss = 0
    best_priors_error_from_r2 = 0
    best_priors_conf_int_from_r2 = 0
    best_priors_error_from_r2w = 0
    best_priors_conf_int_from_r2w = 0
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
        for inputs, targets, weights, dir, view in tqdm(train_loader):  # Simplified unpacking
            inputs, targets, weights = inputs.to('cuda'), targets.to('cuda'), targets.to('cuda')  # Send data to GPU
            # print("inputs, targets, weights, dir, view")
            # print(inputs, "\n", targets, "\n", weights, "\n", dir, "\n", view)
            print("Loaded images\nCurrent GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
            if torch.sum(torch.isnan(inputs)) > 0:
                print("Image is corrupted", torch.sum(torch.isnan(inputs), dim=1))
                nan_mask = torch.isnan(inputs)
                inputs[nan_mask] = 0

            # Zero the parameter gradients
            optimizer.zero_grad()

            is_it_mlo = torch.zeros_like(torch.vstack([targets, targets])).T.float()
            if not split_CC_and_MLO:
                for i in range(len(view)):
                    if 'MLO' in view[i]:
                        is_it_mlo[i][0] += 1
                    else:
                        is_it_mlo[i][1] += 1

            # Forward
            print("Before output\nCurrent GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
            outputs = model.forward(inputs.unsqueeze(1), is_it_mlo)  # Add channel dimension
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

        train_loss /= len(train_loader.dataset)
        scaled_train_loss /= len(train_loader.dataset)

        train_r2 = r2_score(all_targets, all_predictions)
        train_r2w = r2_score(all_targets, all_predictions, sample_weight=np.array(all_targets)+r2_weighting_offset)
        train_err, train_stderr, train_conf = compute_error_metrics(all_targets, all_predictions)
        # Validation
        print("Evaluating on the validation set")
        val_loss, val_labels, val_preds, val_r2, val_r2w, val_err, val_conf = \
            evaluate_model(model, val_loader, criterion,
                           inverse_standardize_targets, mean, std,
                           split_CC_and_MLO=split_CC_and_MLO,
                           r2_weighting_offset=r2_weighting_offset)
        print("Evaluating on the test set")
        test_loss, test_labels, test_preds, test_r2, test_r2w, test_err, test_conf = \
            evaluate_model(model, test_loader, criterion,
                           inverse_standardize_targets, mean, std,
                           split_CC_and_MLO=split_CC_and_MLO,
                           r2_weighting_offset=r2_weighting_offset)
        print("Evaluating on the priors set")
        priors_loss, priors_labels, priors_preds, priors_r2, priors_r2w, priors_err, priors_conf = \
            evaluate_model(model, priors_loader, criterion,
                           inverse_standardize_targets, mean, std,
                           split_CC_and_MLO=split_CC_and_MLO,
                           r2_weighting_offset=r2_weighting_offset)
        val_fig = plot_scatter(val_labels, val_preds, "Validation Scatter Plot " + best_model_name, False, True)
        writer.add_figure("Validation Scatter Plot/{}".format(best_model_name), val_fig, epoch)
        test_fig = plot_scatter(test_labels, test_preds, "Test Scatter Plot " + best_model_name, False, True)
        writer.add_figure("Test Scatter Plot/{}".format(best_model_name), test_fig, epoch)
        priors_fig = plot_scatter(priors_labels, priors_preds, "Priors Scatter Plot " + best_model_name, False, True)
        writer.add_figure("Priors Scatter Plot/{}".format(best_model_name), priors_fig, epoch)
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"\nTrain Loss: {scaled_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Test loss: {test_loss:.4f}, Priors loss: {priors_loss:.4f}"
              f"\nTrain R2: {train_r2:.4f}, Val R2: {val_r2:.4f}, "
              f"Test R2: {test_r2:.4f}, Priors R2: {priors_r2:.4f}"
              f"\nTrain R2w: {train_r2w:.4f}, Val R2w: {val_r2w:.4f}, "
              f"Test R2w: {test_r2w:.4f}, Priors R2w: {priors_r2w:.4f}")

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
        writer.add_scalar('Loss/Priors', priors_loss, epoch)
        writer.add_scalar('R2/Priors', priors_r2, epoch)
        writer.add_scalar('R2 weighted/Priors', priors_r2w, epoch)

        writer.add_scalar('Error/Mean error training', train_err, epoch)
        writer.add_scalar('Error/Confidence interval training', train_conf, epoch)
        writer.add_scalar('Error/Mean error validation', val_err, epoch)
        writer.add_scalar('Error/Confidence interval validation', val_conf, epoch)
        writer.add_scalar('Error/Mean error testing', test_err, epoch)
        writer.add_scalar('Error/Confidence interval testing', test_conf, epoch)
        writer.add_scalar('Error/Mean error priors', priors_err, epoch)
        writer.add_scalar('Error/Confidence interval priors', priors_conf, epoch)

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
            best_priors_r2 = priors_r2
            best_val_r_loss = val_loss
            best_test_r_loss = test_loss
            best_priors_r_loss = priors_loss
            best_val_r_r2w = val_r2w
            best_test_r_r2w = test_r2w
            best_priors_r_r2w = priors_r2w
            best_train_error_from_r2 = train_err
            best_train_conf_int_from_r2 = train_conf
            best_val_error_from_r2 = val_err
            best_val_conf_int_from_r2 = val_conf
            best_test_error_from_r2 = test_err
            best_test_conf_int_from_r2 = test_conf
            best_priors_error_from_r2 = priors_err
            best_priors_conf_int_from_r2 = priors_conf
            not_improved_r2 = 0
            print("Validation R2 improved. Saving best_model.")
            torch.save(model.state_dict(), working_dir + '/../models/rw_' + best_model_name)
        else:
            not_improved_r2 += 1
        if val_r2w > best_val_r2w:
            best_val_r2w = val_r2w
            best_test_r2w = test_r2w
            best_priors_r2w = priors_r2w
            best_val_rw_loss = val_loss
            best_test_rw_loss = test_loss
            best_priors_rw_loss = priors_loss
            best_val_rw_r2 = val_r2
            best_test_rw_r2 = test_r2
            best_priors_rw_r2 = priors_r2
            best_train_error_from_r2w = train_err
            best_train_conf_int_from_r2w = train_conf
            best_val_error_from_r2w = val_err
            best_val_conf_int_from_r2w = val_conf
            best_test_error_from_r2w = test_err
            best_test_conf_int_from_r2w = test_conf
            best_priors_error_from_r2w = priors_err
            best_priors_conf_int_from_r2w = priors_conf
            not_improved_r2w = 0
            print("Validation R2 weighted improved. Saving best_model.")
            torch.save(model.state_dict(), working_dir + '/../models/r_' + best_model_name)
        else:
            not_improved_r2w += 1
        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            best_priors_loss = priors_loss
            best_val_l_r2 = val_r2
            best_test_l_r2 = test_r2
            best_priors_l_r2 = priors_r2
            best_val_l_r2w = val_r2w
            best_test_l_r2w = test_r2w
            best_priors_l_r2w = priors_r2w
            best_train_error_from_loss = train_err
            best_train_conf_int_from_loss = train_conf
            best_val_error_from_loss = val_err
            best_val_conf_int_from_loss = val_conf
            best_test_error_from_loss = test_err
            best_test_conf_int_from_loss = test_conf
            best_priors_error_from_loss = priors_err
            best_priors_conf_int_from_loss = priors_conf
            not_improved_loss = 0
            print("Validation loss improved. Saving best_model.")
            torch.save(model.state_dict(), working_dir + '/../models/l_' + best_model_name)
        else:
            not_improved_loss += 1
        print(f"From best val loss at epoch {epoch - not_improved_loss}:\n "
              f"val loss: {best_val_loss:.4f} test loss {best_test_loss:.4f} priors loss {best_priors_loss:.4f} "
              f"val r2: {best_val_l_r2:.4f} test r2 {best_test_l_r2:.4f} priors r2 {best_priors_l_r2:.4f}"
              f"val r2w: {best_val_l_r2w:.4f} test r2w {best_test_l_r2w:.4f} priors r2w {best_priors_l_r2w:.4f}")
        print(f"From best val R2 at epoch {epoch - not_improved_r2}:\n "
              f"val loss: {best_val_r_loss:.4f} test loss {best_test_r_loss:.4f} priors loss {best_priors_r_loss:.4f} "
              f"val r2: {best_val_r2:.4f} test r2 {best_test_r2:.4f} priors r2 {best_priors_r2:.4f}"
              f"val r2w: {best_val_r_r2w:.4f} test r2w {best_test_r_r2w:.4f} priors r2w {best_priors_r_r2w:.4f}")
        print(f"From best val R2w at epoch {epoch - not_improved_r2w}:\n "
              f"val loss: {best_val_rw_loss:.4f} test loss {best_test_rw_loss:.4f} priors loss {best_priors_rw_loss:.4f} "
              f"val r2: {best_val_rw_r2:.4f} test r2 {best_test_rw_r2:.4f} priors r2 {best_priors_rw_r2:.4f}"
              f"val r2w: {best_val_r2w:.4f} test r2w {best_test_r2w:.4f} priors r2w {best_priors_r2w:.4f}")
        if improving_loss_or_r2 == 'r2':
            time_since_improved = not_improved_r2
        elif improving_loss_or_r2 == 'r2w':
            time_since_improved = not_improved_r2w
        else:
            time_since_improved = not_improved_loss
        if time_since_improved >= patience:
            print("Early stopping")
            break

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
        writer.add_scalar('Loss/Best Priors Loss from Loss', best_priors_loss, epoch)
        writer.add_scalar('Loss/Best Priors Loss from R2', best_priors_r_loss, epoch)
        writer.add_scalar('Loss/Best Priors Loss from R2w', best_priors_rw_loss, epoch)
        writer.add_scalar('R2/Best Priors R2 from R2', best_priors_r2, epoch)
        writer.add_scalar('R2/Best Priors R2 from Loss', best_priors_l_r2, epoch)
        writer.add_scalar('R2/Best Priors R2 from R2w', best_priors_rw_r2, epoch)
        writer.add_scalar('R2w/Best Priors R2w from R2', best_priors_r_r2w, epoch)
        writer.add_scalar('R2w/Best Priors R2w from Loss', best_priors_l_r2w, epoch)
        writer.add_scalar('R2w/Best Priors R2w from R2w', best_priors_r2w, epoch)

        writer.add_scalar('Best Error from loss/Mean error training', best_train_error_from_loss, epoch)
        writer.add_scalar('Best Error from loss/Confidence interval training', best_train_conf_int_from_loss, epoch)
        writer.add_scalar('Best Error from loss/Mean error validation', best_val_error_from_loss, epoch)
        writer.add_scalar('Best Error from loss/Confidence interval validation', best_val_conf_int_from_loss, epoch)
        writer.add_scalar('Best Error from loss/Mean error testing', best_test_error_from_loss, epoch)
        writer.add_scalar('Best Error from loss/Confidence interval testing', best_test_conf_int_from_loss, epoch)
        writer.add_scalar('Best Error from loss/Mean error priors', best_priors_error_from_loss, epoch)
        writer.add_scalar('Best Error from loss/Confidence interval priors', best_priors_conf_int_from_loss, epoch)
        writer.add_scalar('Best Error from r2/Mean error training', best_train_error_from_r2, epoch)
        writer.add_scalar('Best Error from r2/Confidence interval training', best_train_conf_int_from_r2, epoch)
        writer.add_scalar('Best Error from r2/Mean error validation', best_val_error_from_r2, epoch)
        writer.add_scalar('Best Error from r2/Confidence interval validation', best_val_conf_int_from_r2, epoch)
        writer.add_scalar('Best Error from r2/Mean error testing', best_test_error_from_r2, epoch)
        writer.add_scalar('Best Error from r2/Confidence interval testing', best_test_conf_int_from_r2, epoch)
        writer.add_scalar('Best Error from r2/Mean error priors', best_priors_error_from_r2, epoch)
        writer.add_scalar('Best Error from r2/Confidence interval priors', best_priors_conf_int_from_r2, epoch)
        writer.add_scalar('Best Error from r2w/Mean error training', best_train_error_from_r2w, epoch)
        writer.add_scalar('Best Error from r2w/Confidence interval training', best_train_conf_int_from_r2w, epoch)
        writer.add_scalar('Best Error from r2w/Mean error validation', best_val_error_from_r2w, epoch)
        writer.add_scalar('Best Error from r2w/Confidence interval validation', best_val_conf_int_from_r2w, epoch)
        writer.add_scalar('Best Error from r2w/Mean error testing', best_test_error_from_r2w, epoch)
        writer.add_scalar('Best Error from r2w/Confidence interval testing', best_test_conf_int_from_r2w, epoch)
        writer.add_scalar('Best Error from r2w/Mean error priors', best_priors_error_from_r2w, epoch)
        writer.add_scalar('Best Error from r2w/Confidence interval priors', best_priors_conf_int_from_r2w, epoch)

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
    train_loss, train_labels, train_preds, train_r2, train_r2w, train_err, train_conf = \
        evaluate_model(model, train_loader, criterion,
                       inverse_standardize_targets, mean, std,
                       split_CC_and_MLO=split_CC_and_MLO,
                       r2_weighting_offset=r2_weighting_offset)
    print("Final evaluation on the validation set")
    val_loss, val_labels, val_preds, val_r2, val_r2w, val_err, val_conf = \
        evaluate_model(model, val_loader, criterion,
                       inverse_standardize_targets,
                       mean, std, split_CC_and_MLO=split_CC_and_MLO,
                       r2_weighting_offset=r2_weighting_offset)
    print("Final evaluation on the test set")
    test_loss, test_labels, test_preds, test_r2, test_r2w, test_err, test_conf = \
        evaluate_model(model, test_loader, criterion,
                       inverse_standardize_targets,
                       mean, std, split_CC_and_MLO=split_CC_and_MLO,
                       r2_weighting_offset=r2_weighting_offset)
    print("Final evaluation on the priors set")
    priors_loss, priors_labels, priors_preds, priors_r2, priors_r2w, priors_err, priors_conf = \
        evaluate_model(model, priors_loader, criterion,
                       inverse_standardize_targets,
                       mean, std, split_CC_and_MLO=split_CC_and_MLO,
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
    print(f"Priors R2 Score: {priors_r2:.4f}")
    print(f"Priors R2w Score: {priors_r2w:.4f}")
    print(f"Priors Loss: {priors_loss:.4f}")
    print(f"Priors Error: {priors_err:.4f}")
    print(f"Priors Confidence interval: {priors_conf:.4f}")

    # Scatter plots
    plot_scatter(train_labels, train_preds, "Train Scatter Plot " + best_model_name, working_dir + '/results/')
    plot_scatter(val_labels, val_preds, "Validation Scatter Plot " + best_model_name, working_dir + '/results/')
    plot_scatter(test_labels, test_preds, "Test Scatter Plot " + best_model_name, working_dir + '/results/')
    plot_scatter(priors_labels, priors_preds, "Priors Scatter Plot " + best_model_name, working_dir + '/results/')

    # Error distributions
    plot_error_vs_vas(train_labels, train_preds, "Train Error vs VAS " + best_model_name, working_dir + '/results/')
    plot_error_vs_vas(val_labels, val_preds, "Validation Error vs VAS " + best_model_name, working_dir + '/results/')
    plot_error_vs_vas(test_labels, test_preds, "Test Error vs VAS " + best_model_name, working_dir + '/results/')
    plot_error_vs_vas(priors_labels, priors_preds, "Priors Error vs VAS " + best_model_name, working_dir + '/results/')

    # Error distributions
    plot_error_distribution(train_labels, train_preds, "Train Error Distribution " + best_model_name,
                            working_dir + '/results/')
    plot_error_distribution(val_labels, val_preds, "Validation Error Distribution " + best_model_name,
                            working_dir + '/results/')
    plot_error_distribution(test_labels, test_preds, "Test Error Distribution " + best_model_name,
                            working_dir + '/results/')
    plot_error_distribution(priors_labels, priors_preds, "Priors Error Distribution " + best_model_name,
                            working_dir + '/results/')

    print("Done")

    if improving_loss_or_r2 == 'r2':
        return np.max(val_r2)
    else:
        return np.min(val_loss)





if __name__ == "__main__":
    if on_CSF and optuna_optimisation:
        print("Setting up optuna optimisation", time.localtime())
        study_name = '{}_BML_optuna'.format(base_name)  # Unique identifier
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