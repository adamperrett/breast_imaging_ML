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

if on_CSF and not optuna_optimisation:
    config = int(sys.argv[1]) - 1

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

def round_to_(x, sig_fig=2):
   return round(x, -int(floor(log10(abs(x))))+sig_fig)

def regression_training(trial):
    if on_CSF and optuna_optimisation:
        lr = trial.suggest_float('lr', 1e-6, 2e-4, log=True)
        op_choice = 'adam' #trial.suggest_categorical('optimiser', ['adam', 'rms', 'sgd'])#, 'd_adam', 'd_sgd'])
        batch_size = trial.suggest_int('batch_size', 5, 29)
        dropout = trial.suggest_float('dropout', 0, 0.6)
        arch = trial.suggest_categorical('architecture', ['pvas', 'resnetrans'])
        pre_trained = 1 #trial.suggest_categorical('pre_trained', [0, 1])
        replicate = trial.suggest_categorical('replicate', [0, 1])
        transformed = 0 #trial.suggest_categorical('transformed', [0, 1])
        weight_samples = trial.suggest_categorical('weight_samples', [0, 1])
        weight_loss = trial.suggest_categorical('weight_loss', [0, 1])

    best_model_name = '{}_lr{}x{}_{}_p{}r{}_d{}_{}_t{}_wl{}_ws{}'.format(
        base_name, round_to_(lr), batch_size, arch, pre_trained, replicate, round_to_(dropout), op_choice, transformed,
        weight_loss, weight_samples)

    print("Accessing data from", processed_dataset_path, "\nConfig", best_model_name)
    print("Current GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
    train_loader, val_loader, test_loader = return_dataloaders(processed_dataset_path, transformed,
                                                               weight_loss, weight_samples, batch_size)

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

            is_it_mlo = torch.zeros_like(targets).float()
            if not split_CC_and_MLO:
                for i in range(len(view)):
                    if 'MLO' in view[i]:
                        is_it_mlo[i] += 1
                    else:
                        is_it_mlo[i] -= 1

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

        train_r2 = r2_score(all_targets, all_predictions, sample_weight=all_targets)
        # Validation
        print("Evaluating on the validation set")
        val_loss, val_labels, val_preds, val_r2 = evaluate_model(model, val_loader, criterion,
                                                                 inverse_standardize_targets, mean, std,
                                                                 split_CC_and_MLO=split_CC_and_MLO)
        print("Evaluating on the test set")
        test_loss, test_labels, test_preds, test_r2 = evaluate_model(model, test_loader, criterion,
                                                                     inverse_standardize_targets, mean, std,
                                                                     split_CC_and_MLO=split_CC_and_MLO)
        val_fig = plot_scatter(val_labels, val_preds, "Validation Scatter Plot " + best_model_name, False, True)
        writer.add_figure("Validation Scatter Plot/{}".format(best_model_name), val_fig, epoch)
        test_fig = plot_scatter(test_labels, test_preds, "Test Scatter Plot " + best_model_name, False, True)
        writer.add_figure("Test Scatter Plot/{}".format(best_model_name), test_fig, epoch)
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"\nTrain Loss: {scaled_train_loss:.4f}, Val Loss: {val_loss:.4f}, Test loss: {test_loss:.4f}"
              f"\nTrain R2: {train_r2:.4f}, Val R2: {val_r2:.4f}, Test R2: {test_r2:.4f}")

        writer.add_scalar('Loss/Train', scaled_train_loss, epoch)
        writer.add_scalar('R2/Train', train_r2, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('R2/Validation', val_r2, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('R2/Test', test_r2, epoch)

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
            best_val_r_loss = val_loss
            best_test_r_loss = test_loss
            not_improved_r2 = 0
            print("Validation R2 improved. Saving best_model.")
            torch.save(model.state_dict(), working_dir + '/../models/r_' + best_model_name)
        else:
            not_improved_r2 += 1
        # Check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            best_val_l_r2 = val_r2
            best_test_l_r2 = test_r2
            not_improved_loss = 0
            print("Validation loss improved. Saving best_model.")
            torch.save(model.state_dict(), working_dir + '/../models/l_' + best_model_name)
        else:
            not_improved_loss += 1
        print(f"From best val loss at epoch {epoch - not_improved_loss}:\n "
              f"val loss: {best_val_loss:.4f} test loss {best_test_loss:.4f} val r2: {best_val_l_r2:.4f} test r2 {best_test_l_r2:.4f}")
        print(f"From best val R2 at epoch {epoch - not_improved_r2}:\n "
              f"val loss: {best_val_r_loss:.4f} test loss {best_test_r_loss:.4f} val r2: {best_val_r2:.4f} test r2 {best_test_r2:.4f}")
        if improving_loss_or_r2 == 'r2':
            time_since_improved = not_improved_r2
        else:
            time_since_improved = not_improved_loss
        if time_since_improved >= patience:
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
    model.load_state_dict(torch.load(working_dir + '/../models/l_' + best_model_name))

    train_loader.dataset.transform = None
    if weight_samples:
        train_loader.sampler.weights = torch.ones_like(train_loader.sampler.weights)
        train_loader.sampler.replacement = False

    # Evaluating on all datasets: train, val, test
    print("Final evaluation on the train set")
    train_loss, train_labels, train_preds, train_r2 = evaluate_model(model, train_loader, criterion,
                                                                     inverse_standardize_targets, mean, std,
                                                                     split_CC_and_MLO=split_CC_and_MLO)
    print("Final evaluation on the validation set")
    val_loss, val_labels, val_preds, val_r2 = evaluate_model(model, val_loader, criterion, inverse_standardize_targets,
                                                             mean, std, split_CC_and_MLO=split_CC_and_MLO)
    print("Final evaluation on the test set")
    test_loss, test_labels, test_preds, test_r2 = evaluate_model(model, test_loader, criterion, inverse_standardize_targets,
                                                                 mean, std, split_CC_and_MLO=split_CC_and_MLO)

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

    if improving_loss_or_r2 == 'r2':
        return np.max(val_r2)
    else:
        return np.min(val_loss)





if __name__ == "__main__":
    if on_CSF:
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
        print("Beginning optimisation")
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
        print("Running locally")
        _ = regression_training(None)
    print("Done")