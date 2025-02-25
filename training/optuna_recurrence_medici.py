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
from torchmetrics.classification import BinaryAUROC
from sklearn.metrics import accuracy_score

# time.sleep(60*60*14)
print(time.localtime())
seed_value = 27
np.random.seed(seed_value)
torch.set_default_dtype(torch.float32)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sns.set(style='dark')

'''
todo: 
    - make the classification model
- make AUC the test
- proper error calculation
    - processing of data into dataloader
    - optuna params define
- test scripts
'''

def round_to_(x, sig_fig=2):
   return round(x, -int(floor(log10(abs(x))))+sig_fig)

def regression_training(trial):
    global base_name
    lr = trial.suggest_float('lr', 3e-6, 1e-3, log=True)
    op_choice = 'adam' #trial.suggest_categorical('optimiser', ['adam', 'rms', 'sgd'])#, 'd_adam', 'd_sgd'])
    batch_size = 2 #trial.suggest_int('batch_size', 2, 50)
    dropout = trial.suggest_float('dropout', 0, 0.7)
    # arch = trial.suggest_categorical('architecture', ['pvas', 'resnetrans'])
    resnet_size = 18 #trial.suggest_categorical('resent_size', [18, 34, 50])
    pooling_type = 'mean' #trial.suggest_categorical('pooling_type', ['mean', 'max', 'attention'])
    pre_trained = 1 #trial.suggest_categorical('pre_trained', [0, 1])
    include_vas = 1 #trial.suggest_categorical('include_vas', [0, 1])
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
    train_loader, val_loader, test_loader = return_recurrence_loaders(data_path, transformed,
                                                                  weight_loss, weight_samples, batch_size)

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

    recurrence_mapping = [
        'breastrec',
        'distrfievent',
        'local',
        'Ipsbreast',
        'Contrabreast',
    ]

    # Initialize model, criterion, optimizer
    # model = SimpleCNN().to(device)
    #edit cuda
    print("Loading models\nCurrent GPU mem usage is", torch.cuda.memory_allocated() / (1024 ** 2))
    model = Recurrence_MIL_Model(pre_trained, replicate, resnet_size, pooling_type, dropout,
                             split=split_CC_and_MLO, num_manufacturers=num_manufacturers, include_vas=include_vas).to('cuda')
    epsilon = 0.
    # model = TransformerModel(epsilon=epsilon).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')  # BCE for classification
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
    not_improved_auc = 0
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    best_val_auc = -float('inf')
    best_test_auc = -float('inf')
    best_val_l_auc = -float('inf')
    best_test_l_auc = -float('inf')
    best_val_a_loss = float('inf')
    best_test_a_loss = float('inf')
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=int(patience/10), factor=0.9, verbose=True)
    writer = SummaryWriter(working_dir + '/results/' + best_model_name)

    print("Beginning training", time.localtime())
    print("Current GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
    for epoch in tqdm(range(num_epochs)):
        model.train()
        all_preds = [[] for _ in range(5)]
        all_labels = [[] for _ in range(5)]
        train_loss = torch.zeros(5).to('cuda')
        scaled_train_loss = 0.0
        aucs = [BinaryAUROC() for _ in range(5)]
        for image_data, recurrence_data in tqdm(train_loader):  # Simplified unpacking
            # inputs, targets, weights = inputs.to('cuda'), targets.to('cuda'), targets.to('cuda')  # Send data to GPU
            # print("inputs, targets, weights, dir, view")
            # print(inputs, "\n", targets, "\n", weights, "\n", dir, "\n", view)
            # print(f"Loaded images{best_model_name}\nCurrent GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
            # if torch.sum(torch.isnan(image_data)) > 0:
            #     print("Image is corrupted", torch.sum(torch.isnan(image_data), dim=1))
            #     nan_mask = torch.isnan(image_data)
            #     image_data[nan_mask] = 0
            recurrence_data = torch.stack(recurrence_data).to('cuda')

            # Forward
            print("Before output\nCurrent GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
            outputs = model.forward(image_data, manufacturer_mapping)  # Add channel dimension
            print("Before losses\nCurrent GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
            # losses = criterion(outputs.squeeze(1), targets.float())  # Get losses for each sample
            split_losses = []
            for i, t in enumerate(recurrence_data):
                loss = criterion(t.to(torch.float32), outputs[:, i])
                split_losses.append(loss)
            # print("Before weighting\nCurrent GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
            # weighted_loss = (losses * weights).mean()  # Weighted loss

            # Backward + optimize
            print("Before backward\nCurrent GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
            for n, l in enumerate(split_losses):
                if n + 1 < len(split_losses):
                    l.mean().backward(retain_graph=True)
                else:
                    l.mean().backward()
            # weighted_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                total_loss = torch.sum(torch.stack(split_losses), dim=1)
                train_loss += total_loss
                # train_loss += weighted_loss.item() * inputs.size(0)

                print("Before scaling loss\nCurrent GPU mem usage is",  torch.cuda.memory_allocated() / (1024 ** 2))
                for i, auc in enumerate(aucs):
                    auc.update(outputs[:, i], recurrence_data[i])

                for i in range(5):
                    all_preds[i].extend(outputs[:, i].cpu().numpy())
                    all_labels[i].extend(recurrence_data[i].cpu().numpy())
            # with torch.no_grad():
            #     train_outputs_original_scale = outputs.squeeze(1) #inverse_standardize_targets(outputs.squeeze(1), mean, std)
            #     train_targets_original_scale = targets.float() #inverse_standardize_targets(targets.float(), mean, std)
            #     all_targets.extend(train_targets_original_scale.cpu().numpy())
            #     all_predictions.extend(train_outputs_original_scale.cpu().numpy())
            #     scaled_train_loss += criterion(train_outputs_original_scale,
            #                                    train_targets_original_scale).mean().item() * inputs.size(0)
            # break
        train_loss /= len(train_loader.dataset)
        train_auc = torch.stack([auc.compute() for auc in aucs])
        train_acc = []
        train_class = []
        for i in range(5):
            preds_binary = (np.array(all_preds[i]) > 0.5).astype(int)  # Apply threshold for binary classification
            acc = accuracy_score(all_labels[i], preds_binary)
            train_acc.append(acc)
            train_class.append(preds_binary)
        train_acc = torch.tensor(train_acc)
        # Validation
        print("Evaluating on the validation set")
        val_loss, val_labels, val_preds, val_acc, val_class, val_auc, val_manu = \
            evaluate_recurrence(model, val_loader, criterion,
                            inverse_standardize_targets, mean, std,
                            num_manufacturers=num_manufacturers,
                            manufacturer_mapping=manufacturer_mapping,
                            split_CC_and_MLO=split_CC_and_MLO,
                            r2_weighting_offset=r2_weighting_offset
                            )
        print("Evaluating on the test set")
        test_loss, test_labels, test_preds, test_acc, test_class, test_auc, test_manu = \
            evaluate_recurrence(model, test_loader, criterion,
                            inverse_standardize_targets, mean, std,
                            num_manufacturers=num_manufacturers,
                            manufacturer_mapping=manufacturer_mapping,
                            split_CC_and_MLO=split_CC_and_MLO,
                            r2_weighting_offset=r2_weighting_offset
                            )
        print("Evaluating on the pilot set")
        # pilot_loss, pilot_labels, pilot_preds, pilot_r2, pilot_r2w, pilot_err, pilot_conf, pilot_manu = \
        #     evaluate_recurrence(model, pilot_loader, criterion,
        #                     inverse_standardize_targets, mean, std,
        #                     num_manufacturers=num_manufacturers,
        #                     manufacturer_mapping=manufacturer_mapping,
        #                     split_CC_and_MLO=split_CC_and_MLO,
        #                     r2_weighting_offset=r2_weighting_offset)
        val_fig = plot_scatter(val_labels, val_preds, "Validation Scatter Plot " + best_model_name, False, True,
                               manufacturers=val_manu)
        writer.add_figure("Validation Scatter Plot/{}".format(best_model_name), val_fig, epoch)
        test_fig = plot_scatter(test_labels, test_preds, "Test Scatter Plot " + best_model_name, False, True,
                                manufacturers=test_manu)
        writer.add_figure("Test Scatter Plot/{}".format(best_model_name), test_fig, epoch)

        val_fig = plot_auc_curves(val_labels, val_class, "Validation " + best_model_name, recurrence_mapping, False, True)
        writer.add_figure("Validation AUC Plot/{}".format(best_model_name), val_fig, epoch)
        test_fig = plot_auc_curves(test_labels, test_class, "Test " + best_model_name, recurrence_mapping, False, True)
        writer.add_figure("Test AUC Plot/{}".format(best_model_name), test_fig, epoch)
        # pilot_fig = plot_scatter(pilot_labels, pilot_preds, "Pilot Scatter Plot " + best_model_name, False, True,
        #                          manufacturers=pilot_manu)
        # writer.add_figure("Pilot Scatter Plot/{}".format(best_model_name), pilot_fig, epoch)
        print(f"Epoch {epoch + 1}/{num_epochs}, {best_model_name}"
              f"\nTrain Loss: {torch.mean(train_loss):.4f}, Val Loss: {torch.mean(val_loss):.4f}, "
              f"Test loss: {torch.mean(test_loss):.4f}, "
              f"\nTrain auc: {torch.mean(train_auc):.4f} \nVal auc: {torch.mean(val_auc):.4f}"
              f"\nTest auc: {torch.mean(test_auc):.4f}"
              f"\nTrain acc: {torch.mean(train_acc):.4f} \nVal acc: {torch.mean(val_acc):.4f}"
              f"\nTest acc: {torch.mean(test_acc):.4f}"
              )

        writer.add_scalar('Loss/Ave Train', torch.mean(train_loss), epoch)
        writer.add_scalar('Loss/Ave Validation', torch.mean(val_loss), epoch)
        writer.add_scalar('Loss/Ave Test', torch.mean(test_loss), epoch)
        writer.add_scalar('AUC/Ave Train', torch.mean(train_auc), epoch)
        writer.add_scalar('AUC/Ave Validation', torch.mean(val_auc), epoch)
        writer.add_scalar('AUC/Ave Test', torch.mean(test_auc), epoch)
        writer.add_scalar('Acc/Ave Train', torch.mean(train_acc), epoch)
        writer.add_scalar('Acc/Ave Validation', torch.mean(val_acc), epoch)
        writer.add_scalar('Acc/Ave Test', torch.mean(test_acc), epoch)
        for i in range(len(test_auc)):
            current_output_label = recurrence_mapping[i]
            writer.add_scalar('Loss/rec Train {}'.format(current_output_label), train_loss[i], epoch)
            writer.add_scalar('Loss/rec Validation {}'.format(current_output_label), val_loss[i], epoch)
            writer.add_scalar('Loss/rec Test {}'.format(current_output_label), test_loss[i], epoch)
            writer.add_scalar('AUC/rec Train {}'.format(current_output_label), train_auc[i], epoch)
            writer.add_scalar('AUC/rec Validation {}'.format(current_output_label), val_auc[i], epoch)
            writer.add_scalar('AUC/rec Test {}'.format(current_output_label), test_auc[i], epoch)
            writer.add_scalar('Acc/rec Train {}'.format(current_output_label), train_acc[i], epoch)
            writer.add_scalar('Acc/rec Validation {}'.format(current_output_label), val_acc[i], epoch)
            writer.add_scalar('Acc/rec Test {}'.format(current_output_label), test_acc[i], epoch)

        if on_CSF and optuna_optimisation:
            for attempt in range(40):
                try:
                    if improving_loss_or_r2 == 'loss':
                        trial.report(val_loss, epoch)
                    else:
                        trial.report(torch.mean(val_auc), epoch)
                    if trial.should_prune():
                        print("Pruning", best_model_name)
                        raise optuna.TrialPruned()
                except (sqlite3.OperationalError, optuna.exceptions.StorageInternalError) as e:
                    print(f"Attempt {attempt + 1} failed with database lock error: {e}. Retrying in {20} seconds...")
                    time.sleep(20)

        if torch.mean(val_auc) > best_val_auc:
            best_val_a_loss = torch.mean(val_loss)
            best_test_a_loss = torch.mean(test_loss)
            best_val_auc = torch.mean(val_auc)
            best_test_auc = torch.mean(test_auc)
            not_improved_auc = 0
            print("Validation AUC improved. Saving best_model.")
            torch.save(model.state_dict(), working_dir + '/../models/a_' + best_model_name)
        else:
            not_improved_auc += 1
        if torch.mean(val_loss) < best_val_loss:
            best_val_loss = torch.mean(val_loss)
            best_test_loss = torch.mean(test_loss)
            not_improved_loss = 0
            print("Validation Loss improved. Saving best_model.")
            torch.save(model.state_dict(), working_dir + '/../models/l_' + best_model_name)
        else:
            not_improved_loss += 1
        print(f"From best val loss at epoch {epoch - not_improved_loss}:\n "
              f"val loss: {best_val_loss:.4f} test loss {best_test_loss:.4f}"
              f"val auc: {best_val_l_auc:.4f} test auc {best_test_l_auc:.4f}")
        print(f"From best val AUC at epoch {epoch - not_improved_auc}:\n "
              f"val loss: {best_val_a_loss:.4f} test loss {best_test_a_loss:.4f}"
              f"val auc: {best_val_auc:.4f} test auc {best_test_auc:.4f}")
        if improving_loss_or_r2 == 'loss':
            time_since_improved = not_improved_loss
        else:
            time_since_improved = not_improved_auc

        writer.add_scalar('Best Loss/Validation Loss from Loss', best_val_loss, epoch)
        writer.add_scalar('Best Loss/Validation Loss from AUC', best_val_a_loss, epoch)
        writer.add_scalar('Best Loss/Test Loss from Loss', best_test_loss, epoch)
        writer.add_scalar('Best Loss/Test Loss from AUC', best_test_a_loss, epoch)
        writer.add_scalar('Best AUC/Validation AUC from Loss', best_val_l_auc, epoch)
        writer.add_scalar('Best AUC/Validation AUC from AUC', best_val_auc, epoch)
        writer.add_scalar('Best AUC/Test AUC from Loss', best_test_l_auc, epoch)
        writer.add_scalar('Best AUC/Test AUC from AUC', best_test_auc, epoch)

        if time_since_improved >= patience:
            print("Early stopping")
            break
        # scheduler.step(val_loss)

    writer.close()
    print("Loading best model weights!")
    if improving_loss_or_r2 == 'loss':
        model.load_state_dict(torch.load(working_dir + '/../models/l_' + best_model_name))
    else:
        model.load_state_dict(torch.load(working_dir + '/../models/a_' + best_model_name))

    train_loader.dataset.transform = None
    if weight_samples:
        train_loader.sampler.weights = torch.ones_like(train_loader.sampler.weights)
        train_loader.sampler.replacement = False

    # Evaluating on all datasets: train, val, test
    print("Final evaluation on the train set")
    train_loss, train_labels, train_preds, train_acc, train_class, train_auc, train_manu = \
        evaluate_recurrence(model, train_loader, criterion,
                        inverse_standardize_targets, mean, std,
                        num_manufacturers=num_manufacturers,
                        manufacturer_mapping=manufacturer_mapping,
                        split_CC_and_MLO=split_CC_and_MLO,
                        r2_weighting_offset=r2_weighting_offset
                        )
    print("Final evaluation on the validation set")
    val_loss, val_labels, val_preds, val_acc, val_class, val_auc, val_manu = \
        evaluate_recurrence(model, val_loader, criterion,
                        inverse_standardize_targets,
                        mean, std,
                        num_manufacturers=num_manufacturers,
                        manufacturer_mapping=manufacturer_mapping,
                        split_CC_and_MLO=split_CC_and_MLO,
                        r2_weighting_offset=r2_weighting_offset
                        )
    print("Final evaluation on the test set")
    test_loss, test_labels, test_preds, test_acc, test_class, test_auc, test_manu = \
        evaluate_recurrence(model, test_loader, criterion,
                        inverse_standardize_targets,
                        mean, std,
                        num_manufacturers=num_manufacturers,
                        manufacturer_mapping=manufacturer_mapping,
                        split_CC_and_MLO=split_CC_and_MLO,
                        r2_weighting_offset=r2_weighting_offset
                        )
    # print("Final evaluation on the pilot set")
    # pilot_loss, pilot_labels, pilot_preds, pilot_r2, pilot_r2w, pilot_err, pilot_conf, pilot_manu = \
    #     evaluate_recurrence(model, pilot_loader, criterion,
    #                     inverse_standardize_targets,
    #                     mean, std,
    #                     num_manufacturers=num_manufacturers,
    #                     manufacturer_mapping=manufacturer_mapping,
    #                     split_CC_and_MLO=split_CC_and_MLO,
    #                     r2_weighting_offset=r2_weighting_offset)

    # R2 Scores
    print(f"Train Loss: {torch.mean(train_loss):.4f}")
    print(f"Train AUC: {torch.mean(train_auc):.4f}")
    print(f"Train Acc: {torch.mean(train_acc):.4f}")
    print(f"Validation Loss: {torch.mean(val_loss):.4f}")
    print(f"Validation AUC: {torch.mean(val_auc):.4f}")
    print(f"Validation Acc: {torch.mean(val_acc):.4f}")
    print(f"Test Loss: {torch.mean(test_loss):.4f}")
    print(f"Test AUC: {torch.mean(test_auc):.4f}")
    print(f"Test Acc: {torch.mean(test_acc):.4f}")

    # Scatter plots
    plot_scatter(train_labels, train_preds, "Train Scatter Plot " + best_model_name, working_dir + '/results/',
                 manufacturers=train_manu)
    plot_scatter(val_labels, val_preds, "Validation Scatter Plot " + best_model_name, working_dir + '/results/',
                 manufacturers=val_manu)
    plot_scatter(test_labels, test_preds, "Test Scatter Plot " + best_model_name, working_dir + '/results/',
                 manufacturers=test_manu)
    # plot_scatter(pilot_labels, pilot_preds, "Pilot Scatter Plot " + best_model_name, working_dir + '/results/',
    #              manufacturers=pilot_manu)

    # Error distributions
    plot_auc_curves(train_labels, train_class, "Train " + best_model_name, recurrence_mapping, working_dir + '/results/')
    plot_auc_curves(val_labels, val_class, "Validation " + best_model_name, recurrence_mapping, working_dir + '/results/')
    plot_auc_curves(test_labels, test_class, "Test " + best_model_name, recurrence_mapping, working_dir + '/results/')

    print("Done")

    if improving_loss_or_r2 == 'loss':
        return val_loss
    else:
        return torch.mean(val_auc)





if __name__ == "__main__":
    if on_CSF and optuna_optimisation:
        print("Setting up optuna optimisation", time.localtime())
        study_name = '{}_recurrence_optuna'.format(base_name)  # Unique identifier
        storage_url = 'sqlite:///{}.db'.format(study_name)
        if improving_loss_or_r2 == 'loss':
            direction = 'minimize'
        else:
            direction = 'maximize'
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