import random
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pad
from torchvision import transforms
import os
from torch.utils.data import Dataset, Subset, WeightedRandomSampler
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
parent_dir = os.path.dirname(current_path)
sys.path.insert(0, parent_dir)
from training.training_config import *
from data_processing.data_analysis import *
import time
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

gpu = True
if gpu:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = 'cuda'
else:
    torch.set_default_tensor_type(torch.FloatTensor)
    device = 'cpu'


class RecurrenceLoader(Dataset):
    def __init__(self, dataset, transform=None, by_patient_or_by_image='patient', weights=None):
        self.transform = transform
        self.by_patient_or_by_image = by_patient_or_by_image
        self.weights = weights
        self.dataset = []
        for patient in dataset:
            patient_data = []
            failed = False
            for timepoint in dataset[patient]:
                if timepoint == 'recurrence':
                    patient_data.append(dataset[patient][timepoint])
                else:
                    if dataset[patient][timepoint]['failed']:
                        failed = True
                        break
                    mlo_image = dataset[patient][timepoint]['mlo'][0].to(torch.float32)
                    cc_image = dataset[patient][timepoint]['cc'][0].to(torch.float32)
                    score = dataset[patient][timepoint]['score']
                    manu = dataset[patient][timepoint]['manufacturer']
                    patient_data.append([
                        mlo_image,
                        score,
                        # score,  # weight
                        timepoint,
                        patient,
                        manu,
                        'mlo'
                    ])
                    patient_data.append([
                        cc_image,
                        score,
                        # score,  # weight
                        timepoint,
                        patient,
                        manu,
                        'cc'
                    ])
            if not failed:
                self.dataset.append(patient_data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if idx >= len(self.dataset):
            print(f"\nBad idx: {idx}, dataset length: {len(self.dataset)}\n")
        patient = self.dataset[idx]
        recurrence_data = list(patient[0].values())
        image_data = patient[1:]
        transformed_image_data = []
        for image, score, timepoint, patient, manu, view in image_data:
            if self.transform:
                transformed_image = self.transform(image)
                image = transformed_image
            transformed_image_data.append([image, score, timepoint, patient, manu, view])

        return transformed_image_data, recurrence_data

class MediciLoader(Dataset):
    def __init__(self, dataset, transform=None, by_patient_or_by_image='patient', weights=None):
        self.transform = transform
        self.by_patient_or_by_image = by_patient_or_by_image
        self.weights = weights
        if self.by_patient_or_by_image == 'image':
            self.dataset = []
            for patient in dataset:
                for timepoint in dataset[patient]:
                    if not dataset[patient][timepoint]['failed']:
                        image = dataset[patient][timepoint]['mlo'][0].to(torch.float32)
                        score = dataset[patient][timepoint]['score']
                        manu = dataset[patient][timepoint]['manufacturer']
                        view = 'mlo'
                        self.dataset.append([
                            image,
                            score,
                            # score,  # weight by score for now
                            timepoint,
                            patient,
                            manu,
                            view
                        ])
                        image = dataset[patient][timepoint]['cc'][0].to(torch.float32)
                        view = 'cc'
                        self.dataset.append([
                            image,
                            score,
                            # score,  # weight
                            timepoint,
                            patient,
                            manu,
                            view
                        ])
        else:
            self.dataset = []
            for patient in dataset:
                for timepoint in dataset[patient]:
                    if not dataset[patient][timepoint]['failed']:
                        mlo_image = dataset[patient][timepoint]['mlo'][0].to(torch.float32)
                        cc_image = dataset[patient][timepoint]['cc'][0].to(torch.float32)
                        images = torch.stack([mlo_image, cc_image])
                        score = dataset[patient][timepoint]['score']
                        manu = dataset[patient][timepoint]['manufacturer']
                        views = ['mlo', 'cc']
                        self.dataset.append([
                            images,
                            score,
                            # score,  # weight
                            timepoint,
                            patient,
                            manu,
                            views
                        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        patient = self.dataset[idx]
        image, label, weight, patient, manu, views = patient
        if self.transform:
            if self.by_patient_or_by_image == 'patient':
                transformed_image = self.transform(image)
            else:
                transformed_image = [self.transform(im.unsqueeze(0)).squeeze(0) for im in image]
            image = transformed_image

        return image, label, weight, patient, manu, views


class MosaicEvaluateLoader(Dataset):
    def __init__(self, dataset, transform=None, max_n=1, weights=None, rand_select=True):
        self.dataset = dataset
        self.transform = transform
        self.max_n = max_n
        self.weights = weights
        self.rand_select = rand_select

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, _, _, dir, views = self.dataset[idx]
        sample = range(len(image))
        if self.transform:
            transformed_image = [self.transform(im.unsqueeze(0)).squeeze(0) for im in image[sample]]
            image = transformed_image
        else:
            image = image[sample]
        new_views = [views[s] for s in sample]

        # If weights are provided, return them as well
        if self.weights is not None:
            return image, label, self.weights[idx], dir, new_views
        else:
            return image, label, 1, dir, new_views


class MosaicDataset(Dataset):
    def __init__(self, dataset, transform=None, max_n=4, weights=None, rand_select=True):
        self.dataset = dataset
        self.transform = transform
        self.max_n = max_n
        self.weights = weights
        self.rand_select = rand_select

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label, _, _, dir, views = self.dataset[idx]
        if len(image) > self.max_n:
            sample = random.sample(range(len(image)), self.max_n)
        else:
            if len(image) < self.max_n:
                for i in range(self.max_n - len(image)):
                    image = torch.vstack([image, torch.zeros_like(image[0]).unsqueeze(0)])
                    views.append(views[np.random.randint(len(views)-i)])
            sample = range(self.max_n)
        if self.transform:
            transformed_image = [self.transform(im.unsqueeze(0)).squeeze(0) for im in image[sample]]
            image = transformed_image
        else:
            image = image[sample]
        new_views = [views[s] for s in sample]

        # If weights are provided, return them as well
        if self.weights is not None:
            return image, label, self.weights[idx], dir, new_views
        else:
            return image, label, 1, dir, new_views


class MammogramDataset(Dataset):
    def __init__(self, dataset, transform=None, n=0, weights=None, rand_select=True, no_process=False):
        self.dataset = dataset #torch.load(dataset_path)
        self.transform = transform
        self.n = n
        self.weights = weights
        self.rand_select = rand_select
        self.no_process = no_process

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.no_process:
            return self.dataset[idx]
        image, label, r1, r2, directory, views = self.dataset[idx]

        if self.transform:
            transformed_image = self.transform(image.unsqueeze(0)).squeeze(0)
            image = transformed_image

        if self.weights is not None:
            return image, label, self.weights[idx], directory, views
        else:
            return image, label, (label*0)+1, directory, views


class MammogramThresholdedDataset(Dataset):
    def __init__(self, dataset, transform=None, n=0, weights=None, rand_select=True, no_process=False, threshold=100):
        self.dataset = []
        for image, label, r1, r2, directory, views in dataset:
            if abs(r1 - r2) <= threshold:
                self.dataset.append(image, label, directory, views)
        self.transform = transform
        self.n = n
        self.weights = weights
        self.rand_select = rand_select
        self.no_process = no_process

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.no_process:
            return self.dataset[idx]
        image, label, directory, views = self.dataset[idx]

        if self.transform:
            transformed_image = self.transform(image.unsqueeze(0)).squeeze(0)
            image = transformed_image

        if self.weights is not None:
            return image, label, self.weights[idx], directory, views
        else:
            return image, label, (label*0)+1, directory, views

def custom_collate(batch):
    # Separate images and labels
    images, labels, weights, dir, view = zip(*batch)

    # Determine the max combined width
    max_width = max([sum(img.size(-1) for img in img_list) for img_list in images])

    # Stack images horizontally with padding
    stacked_images = []
    for img_list in images:
        combined_width = sum(img.size(-1) for img in img_list)
        padding_size = max_width - combined_width
        combined_img = torch.cat(tuple(img_list), dim=-1)  # Use tuple() here
        if padding_size > 0:
            # Pad on the right
            combined_img = pad(combined_img, (0, 0, padding_size, 0))
        stacked_images.append(combined_img)

    # Stack the processed images into a batch
    images_tensor = torch.stack(stacked_images)

    # Convert the list of regression targets to a tensor and standardize
    labels_tensor = torch.tensor(labels, dtype=torch.float32)  # Change the dtype if needed
    labels_tensor = standardize_targets(labels_tensor, mean, std)

    # if weights[0]:
    return images_tensor, labels_tensor, torch.tensor(weights, dtype=torch.float32)


print("Reading data")

mean = 0
std = 0

def split_by_patient_and_stratefy_by_manufacturer(dataset_path, train_ratio, val_ratio, test_ratio, seed_value=0):
    print("Loading data to split by patient", time.localtime())
    dataset = torch.load(dataset_path)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Grouping entries by the unique key", time.localtime())
    patient_manufacturers = []
    patient_list = []
    manufactorer_index = {}
    for patient in tqdm(dataset):
        if len(dataset[patient]) != 4:
            continue
        patient_manufacturers.append([])
        patient_list.append(patient)
        for timepoint in dataset[patient]:
            if timepoint == 'recurrence':
                new_cancer = dataset[patient][timepoint]['Ipsbreast'] or dataset[patient][timepoint]['Contrabreast']
                dataset[patient][timepoint]['new_cancer'] = new_cancer
                del dataset[patient][timepoint]['Ipsbreast']
                del dataset[patient][timepoint]['Contrabreast']
                manufacturer = '{}'.format(dataset[patient][timepoint].values())
            else:
                manufacturer = dataset[patient][timepoint]['manufacturer']
            if manufacturer not in manufactorer_index:
                manufactorer_index[manufacturer] = len(manufactorer_index)
            patient_manufacturers[-1].append(manufactorer_index[manufacturer])
        patient_manufacturers[-1] = np.array(patient_manufacturers[-1])
    patient_list = np.array(patient_list)
    patient_manufacturers = np.array(patient_manufacturers)

    parsed_manufactorers = []
    for manu in patient_manufacturers:
        zeros = [0 for i in range(len(manufactorer_index))]
        for m in manu:
            zeros[m] = 1
        parsed_manufactorers.append(np.array(zeros))
    parsed_manufactorers = np.array(parsed_manufactorers)

    splitter1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1-train_ratio,
                                                 random_state=seed_value)
    train_index, temp_index = next(splitter1.split(patient_list, parsed_manufactorers))
    training_patients, val_test_patients = patient_list[train_index], patient_list[temp_index]
    training_manufacturer, val_test_manufacturer = patient_manufacturers[train_index], patient_manufacturers[temp_index]

    splitter1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio/(val_ratio+test_ratio),
                                                 random_state=seed_value)
    val_index, test_index = next(splitter1.split(patient_list[temp_index], parsed_manufactorers[temp_index]))
    val_patients, val_manufacturer = val_test_patients[val_index], val_test_manufacturer[val_index]
    test_patients, test_manufacturer = val_test_patients[test_index], val_test_manufacturer[test_index]

    # Split the patient data into train, validation, and test sets
    train_data = {tp: dataset[tp] for tp in training_patients}
    val_data = {vp: dataset[vp] for vp in val_patients}
    test_data = {tp: dataset[tp] for tp in test_patients}
    print("Splits across manufacturer:")
    print("Training", {cl:np.sum(np.sum([[a == cl for a in b] for b in training_manufacturer]))
                       for cl in range(len(manufactorer_index))})
    print("Validation", {cl:np.sum([[a == cl for a in b] for b in val_manufacturer])
                       for cl in range(len(manufactorer_index))})
    print("Testing", {cl:np.sum([[a == cl for a in b] for b in test_manufacturer])
                       for cl in range(len(manufactorer_index))})

    return train_data, val_data, test_data

def split_and_group_by_patient(dataset_path, train_ratio, val_ratio, seed_value=0):
    print("Loading data to split by patient", time.localtime())
    dataset = torch.load(dataset_path)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Grouping entries by the unique key", time.localtime())
    groups = {}
    for entry in tqdm(dataset):
        key = entry[-2]  # The second last entry where the patient id is
        if key not in groups:
            groups[key] = []
        groups[key].append(entry)

    # Convert groups to a list of patient data, each containing all images and averaged target for that patient
    patient_data = []
    for key, entries in groups.items():
        images = torch.stack([entry[0] for entry in entries])  # Assuming the first element is the image tensor
        targets = [entry[1] for entry in entries]  # Assuming the second element is the target
        r1 = [entry[2] for entry in entries]  # Assuming the second element is the target
        r2 = [entry[3] for entry in entries]  # Assuming the second element is the target
        dir = [entry[4] for entry in entries]  # Assuming the second element is the target
        dir = dir[0]
        view = [entry[-1] for entry in entries]  # last element is the view
        avg_target = sum(targets) / len(targets)
        patient_data.append((images, avg_target, r1, r2, dir, view))

    # Shuffle the patient data for randomness
    random.shuffle(patient_data)

    # Calculate the split sizes based on the number of patients
    train_end = int(len(patient_data) * train_ratio)
    val_end = train_end + int(len(patient_data) * val_ratio)

    # Split the patient data into train, validation, and test sets
    train_data = patient_data[:train_end]
    val_data = patient_data[train_end:val_end]
    test_data = patient_data[val_end:]

    return train_data, val_data, test_data

def split_into_groups(data, m):
    """
    Split a list or tensor into m even groups.

    Args:
        data (list or tensor): The data to split.
        m (int): The number of groups.

    Returns:
        List of tensors: Groups of approximately equal size.
    """
    n = len(data)
    group_size = n // m
    remainder = n % m  # Remaining items to distribute

    groups = []
    start_idx = 0

    for i in range(m):
        # Calculate the size of the current group
        extra = 1 if i < remainder else 0
        end_idx = start_idx + group_size + extra

        # Append the current group
        groups.append(data[start_idx:end_idx])
        start_idx = end_idx

    return groups


def split_by_patient_and_current_crossval(dataset_path, train_ratio, val_ratio, test_ratio, seed_value=0, current_fold=0):
    print("Loading data to split by patient", time.localtime())
    dataset = torch.load(dataset_path)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Grouping entries by the unique key", time.localtime())
    patient_manufacturers = []
    patient_list = []
    manufactorer_index = {}
    for patient in dataset:
        patient_list.append(patient)
    patient_list = np.array(patient_list)

    n_numbers = torch.randperm(len(patient_list))  # Numbers from 0 to 100
    m_groups = int(1 / 0.05)  # Number of groups

    groups = split_into_groups(n_numbers, m_groups)
    shift = current_fold * 2
    train_group_indexes = (torch.linspace(0, 14, 15, dtype=torch.int16) + shift) % m_groups
    val_group_indexes = (torch.linspace(15, 17, 3, dtype=torch.int16) + shift) % m_groups
    test_group_indexes = (torch.linspace(18, 19, 2, dtype=torch.int16) + shift) % m_groups
    train_index = torch.hstack([groups[idx] for idx in train_group_indexes])
    val_index = torch.hstack([groups[idx] for idx in val_group_indexes])
    test_index = torch.hstack([groups[idx] for idx in test_group_indexes])
    training_patients = patient_list[train_index]
    val_patients = patient_list[val_index]
    test_patients = patient_list[test_index]

    # Split the patient data into train, validation, and test sets
    train_data = {tp: dataset[tp] for tp in training_patients}
    val_data = {vp: dataset[vp] for vp in val_patients}
    test_data = {tp: dataset[tp] for tp in test_patients}

    return train_data, val_data, test_data

def split_by_patient(dataset_path, train_ratio, val_ratio, seed_value=0):
    print("Loading data to split by patient", time.localtime())
    dataset = torch.load(dataset_path)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("Grouping entries by the unique key", time.localtime())
    groups = {}
    for entry in tqdm(dataset):
        key = entry[-2]  # The second last entry where the patient id is
        if key not in groups:
            groups[key] = []
        groups[key].append(entry)

    # Shuffle the list of keys for randomness
    keys = list(groups.keys())
    random.shuffle(keys)

    # Calculate the split sizes based on the number of unique keys
    train_end = int(len(keys) * train_ratio)
    val_end = train_end + int(len(keys) * val_ratio)

    # Split the keys into train, validation, and test
    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]

    # Select data from the original dataset based on the split keys
    train_data = [item for k in train_keys for item in groups[k]]
    val_data = [item for k in val_keys for item in groups[k]]
    test_data = [item for k in test_keys for item in groups[k]]

    return train_data, val_data, test_data

def return_dataloaders(file_name, transformed, weighted_loss, weighted_sampling, batch_size, seed_value=0,
                       only_testing=False):
    print("Beginning data loading", time.localtime())

    full_processed_data_address = os.path.join(processed_dataset_path, file_name+'.pth')
    if only_testing:
        print(f"Loading data {file_name} for testing from {processed_dataset_path}")
        print(time.localtime())
        data = torch.load(full_processed_data_address)
        dataset = MammogramDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            generator=torch.Generator(device=device))
        return loader

    global mean, std

    print(f"Data being collected = {file_name} from {processed_dataset_path}")
    print(time.localtime())
    save_path = os.path.join(working_dir, file_name + '_data.pth')
    if os.path.exists(save_path):
        print("Loading data", time.localtime())
        data = torch.load(save_path)
        train_data, val_data, test_data = data['train'], data['val'], data['test']
        mean, std = data['mean'], data['std']
        computed_weights = data['weights'] if weighted_sampling else None
    else:
        print("Processing data for the first time", time.localtime())
        # Splitting the dataset
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
        train_data, val_data, test_data = split_by_patient(full_processed_data_address,
                                                           train_ratio, val_ratio, seed_value)

        # Compute weights for the training set
        targets = [label for _, label, _, _, _, _ in train_data]
        computed_weights = targets  # compute_sample_weights(targets)

        mean, std = compute_target_statistics(targets)

        print("Saving data", time.localtime())
        torch.save({
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'mean': mean,
            'std': std,
            'weights': computed_weights
        },
            save_path)

    if weighted_sampling:
        sample_weights = computed_weights
    else:
        sample_weights = None

    if transformed:
        # Define your augmentations
        data_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            # Assuming images are PIL images; if not, you'll need to adjust or implement suitable transformations
        ])
    else:
        data_transforms = None

    # Create Dataset
    print("Creating Dataset", time.localtime())
    val_dataset = MammogramDataset(val_data)
    if optuna_optimisation:
        train_dataset = MammogramDataset(train_data, transform=data_transforms, weights=sample_weights)
        test_dataset = MammogramDataset(test_data)
    else:
        train_data.extend(test_data)
        train_dataset = MammogramDataset(train_data, transform=data_transforms, weights=sample_weights)
        test_dataset = MammogramDataset(test_data[-10:])

    # Create DataLoaders
    print("Creating DataLoaders for", device, time.localtime())
    if by_patient:
        if weighted_sampling:
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=WeightedRandomSampler(weights=sample_weights,
                                                                    num_samples=len(train_dataset),
                                                                    replacement=True),
                                      collate_fn=custom_collate,
                                      generator=torch.Generator(device=device))
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate,
                                      generator=torch.Generator(device=device))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate,
                                generator=torch.Generator(device=device))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate,
                                 generator=torch.Generator(device=device))
    else:
        if weighted_sampling:
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=WeightedRandomSampler(weights=sample_weights,
                                                                    num_samples=len(train_dataset),
                                                                    replacement=True),
                                      generator=torch.Generator(device=device))
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      generator=torch.Generator(device=device))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                generator=torch.Generator(device=device))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 generator=torch.Generator(device=device))

    return train_loader, val_loader, test_loader

def return_mosaic_loaders(file_name, transformed, weighted_loss, weighted_sampling, batch_size, seed_value=0,
                       only_testing=False):
    print("Beginning data loading", time.localtime())

    full_processed_data_address = os.path.join(processed_dataset_path, file_name+'.pth')
    if only_testing:
        print(f"Loading data {file_name} for testing from {processed_dataset_path}")
        print(time.localtime())
        train_data, _, _ = split_and_group_by_patient(full_processed_data_address,
                                                           1., 0, seed_value)
        data = train_data
        dataset = MosaicEvaluateLoader(data, max_n=4)
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            generator=torch.Generator(device=device))
        return loader

    global mean, std

    print(f"Data being collected = {file_name} from {processed_dataset_path}")
    print(time.localtime())
    save_path = os.path.join(working_dir, file_name + '_data.pth')
    if os.path.exists(save_path):
        print("Loading data", time.localtime())
        data = torch.load(save_path)
        train_data, val_data, test_data = data['train'], data['val'], data['test']
        mean, std = data['mean'], data['std']
        computed_weights = data['weights'] if weighted_sampling else None
    else:
        print("Processing data for the first time", time.localtime())
        # Splitting the dataset
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        train_data, val_data, test_data = split_by_patient(full_processed_data_address,
                                                           train_ratio, val_ratio, seed_value)

        # Compute weights for the training set
        targets = [label for _, label, _, _, _, _ in train_data]
        computed_weights = targets  # compute_sample_weights(targets)

        mean, std = compute_target_statistics(targets)

        print("Saving data", time.localtime())
        torch.save({
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'mean': mean,
            'std': std,
            'weights': computed_weights
        },
            save_path)

    if weighted_sampling:
        sample_weights = computed_weights
    else:
        sample_weights = None

    if transformed:
        # Define your augmentations
        data_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            # Assuming images are PIL images; if not, you'll need to adjust or implement suitable transformations
        ])
    else:
        data_transforms = None

    # Create Dataset
    print("Creating Dataset", time.localtime())
    val_dataset = MosaicEvaluateLoader(val_data, max_n=4)
    if optuna_optimisation:
        train_dataset = MosaicDataset(train_data, transform=data_transforms, weights=sample_weights)
        test_dataset = MosaicEvaluateLoader(test_data, max_n=4)
    else:
        train_data.extend(test_data)
        train_dataset = MosaicDataset(train_data, transform=data_transforms, weights=sample_weights)
        test_dataset = MosaicEvaluateLoader(test_data[-10:])

    # Create DataLoaders
    print("Creating DataLoaders for", device, time.localtime())
    if by_patient:
        if weighted_sampling:
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=WeightedRandomSampler(weights=sample_weights,
                                                                    num_samples=len(train_dataset),
                                                                    replacement=True),
                                      collate_fn=custom_collate,
                                      generator=torch.Generator(device=device))
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate,
                                      generator=torch.Generator(device=device))
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate,
                                generator=torch.Generator(device=device))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate,
                                 generator=torch.Generator(device=device))
    else:
        if weighted_sampling:
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=WeightedRandomSampler(weights=sample_weights,
                                                                    num_samples=len(train_dataset),
                                                                    replacement=True),
                                      generator=torch.Generator(device=device))
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      generator=torch.Generator(device=device))
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                generator=torch.Generator(device=device))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 generator=torch.Generator(device=device))

    return train_loader, val_loader, test_loader

def return_medici_loaders(file_name, transformed, weighted_loss, weighted_sampling, batch_size, seed_value=0,
                       only_testing=False):
    print("Beginning data loading", time.localtime())

    full_processed_data_address = os.path.join(processed_dataset_path, file_name+'.pth')
    if only_testing:
        print(f"Loading data {file_name} for testing from {processed_dataset_path}")
        print(time.localtime())
        train_data = torch.load(full_processed_data_address)
        data = train_data
        dataset = MediciLoader(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            generator=torch.Generator(device=device))
        return loader

    global mean, std

    print(f"Data being collected = {file_name} from {processed_dataset_path}")
    print(time.localtime())
    save_path = os.path.join(working_dir, file_name + '_data.pth')
    if os.path.exists(save_path):
        print("Loading data", time.localtime())
        data = torch.load(save_path)
        train_data, val_data, test_data = data['train'], data['val'], data['test']
        mean, std = data['mean'], data['std']
        computed_weights = data['weights'] if weighted_sampling else None
    else:
        print("Processing data for the first time", time.localtime())
        # Splitting the dataset
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
        train_data, val_data, test_data = split_by_patient_and_stratefy_by_manufacturer(
            full_processed_data_address,
            train_ratio, val_ratio, test_ratio, seed_value)

        # Compute weights for the training set
        targets = np.hstack([
            [train_data[patient][0]['score'] for patient in train_data
             if 0 in train_data[patient] and not train_data[patient][0]['failed']],
            [train_data[patient][1]['score'] for patient in train_data
             if 1 in train_data[patient] and not train_data[patient][1]['failed']],
            [train_data[patient][3]['score'] for patient in train_data
             if 3 in train_data[patient] and not train_data[patient][3]['failed']]])
        computed_weights = targets  # compute_sample_weights(targets)

        mean, std = compute_target_statistics(targets)

        print("Saving data", time.localtime())
        torch.save({
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'mean': mean,
            'std': std,
            'weights': computed_weights
        },
            save_path)

    if weighted_sampling:
        sample_weights = computed_weights
    else:
        sample_weights = None

    if transformed:
        # Define your augmentations
        data_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            # Assuming images are PIL images; if not, you'll need to adjust or implement suitable transformations
        ])
    else:
        data_transforms = None

    # Create Dataset
    print("Creating Dataset", time.localtime())
    train_dataset = MediciLoader(train_data, transform=data_transforms, weights=sample_weights)
    val_dataset = MediciLoader(val_data)
    test_dataset = MediciLoader(test_data)

    # Create DataLoaders
    print("Creating DataLoaders for", device, time.localtime())
    if by_patient:
        if weighted_sampling:
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=WeightedRandomSampler(weights=sample_weights,
                                                                    num_samples=len(train_dataset),
                                                                    replacement=True),
                                      collate_fn=custom_collate,
                                      generator=torch.Generator(device=device))
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate,
                                      generator=torch.Generator(device=device))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate,
                                generator=torch.Generator(device=device))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate,
                                 generator=torch.Generator(device=device))
    else:
        if weighted_sampling:
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=WeightedRandomSampler(weights=sample_weights,
                                                                    num_samples=len(train_dataset),
                                                                    replacement=True),
                                      generator=torch.Generator(device=device))
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      generator=torch.Generator(device=device))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                generator=torch.Generator(device=device))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 generator=torch.Generator(device=device))

    return train_loader, val_loader, test_loader

def return_recurrence_loaders(file_name, transformed, weighted_loss, weighted_sampling, batch_size, seed_value=0,
                       only_testing=False):
    print("Beginning data loading", time.localtime())

    full_processed_data_address = os.path.join(processed_dataset_path, file_name+'.pth')

    global mean, std

    print("Processing data for the first time", time.localtime())
    # Splitting the dataset
    train_ratio, val_ratio, test_ratio = 0.5, 0.25, 0.25
    train_data, val_data, test_data = split_by_patient_and_stratefy_by_manufacturer(
        full_processed_data_address,
        train_ratio, val_ratio, test_ratio, seed_value)

    # Compute weights for the training set
    targets = np.hstack([
        [train_data[patient][0]['score'] for patient in train_data
         if 0 in train_data[patient] and not train_data[patient][0]['failed']],
        [train_data[patient][1]['score'] for patient in train_data
         if 1 in train_data[patient] and not train_data[patient][1]['failed']],
        [train_data[patient][3]['score'] for patient in train_data
         if 3 in train_data[patient] and not train_data[patient][3]['failed']]])
    classes_0 = np.hstack(
        [train_data[patient]['recurrence']['breastrec'] for patient in train_data
         if 0 in train_data[patient] and not train_data[patient][0]['failed']])
    classes_013 = np.hstack(
        [train_data[patient]['recurrence']['breastrec'] for patient in train_data
         if 0 in train_data[patient] and not train_data[patient][0]['failed']
         and 1 in train_data[patient] and not train_data[patient][1]['failed']
         and 3 in train_data[patient] and not train_data[patient][3]['failed']])
    classes = np.hstack(
        [train_data[patient]['recurrence']['breastrec'] for patient in train_data])
    computed_weights = np.abs(classes_013 - 0.03)  # compute_sample_weights(targets)

    mean, std = compute_target_statistics(targets)

    if weighted_sampling:
        if weighted_sampling == 1:
            sample_weights = computed_weights
        elif weighted_sampling == 2:
            sample_weights = 1 - computed_weights
        else:
            sample_weights = (computed_weights * 0) + 0.5
        print("\n\n sw {} td {} c {} c0 {} c013 {}\n\n".format(len(sample_weights), len(train_data), len(classes),
                                                               len(classes_0), len(classes_013)))
    else:
        sample_weights = None

    if transformed:
        # Define your augmentations
        data_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            # Assuming images are PIL images; if not, you'll need to adjust or implement suitable transformations
        ])
    else:
        data_transforms = None

    # Create Dataset
    print("Creating Dataset", time.localtime())
    train_dataset = RecurrenceLoader(train_data, transform=data_transforms, weights=sample_weights)
    val_dataset = RecurrenceLoader(val_data)
    test_dataset = RecurrenceLoader(test_data)

    # Create DataLoaders
    print("Creating DataLoaders for", device, time.localtime())
    if by_patient:
        if weighted_sampling:
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=WeightedRandomSampler(weights=sample_weights,
                                                                    num_samples=len(sample_weights),
                                                                    replacement=True),
                                      collate_fn=custom_collate,
                                      generator=torch.Generator(device=device), drop_last=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate,
                                      generator=torch.Generator(device=device), drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate,
                                generator=torch.Generator(device=device), drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate,
                                 generator=torch.Generator(device=device), drop_last=True)
    else:
        if weighted_sampling:
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=WeightedRandomSampler(weights=sample_weights,
                                                                    num_samples=len(sample_weights),
                                                                    replacement=True),
                                      generator=torch.Generator(device=device), drop_last=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      generator=torch.Generator(device=device), drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                generator=torch.Generator(device=device), drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 generator=torch.Generator(device=device), drop_last=True)

    return train_loader, val_loader, test_loader

def return_crossval_loaders(file_name, transformed, weighted_loss, weighted_sampling, batch_size,
                       only_testing=False):
    print("Beginning data loading", time.localtime())

    full_processed_data_address = os.path.join(processed_dataset_path, file_name+'.pth')

    global mean, std

    print(f"Data being collected = {file_name} from {processed_dataset_path}")
    print(time.localtime())
    print("Processing data for the first time", time.localtime())
    # Splitting the dataset
    cross_fold_csv_filename = 'all_cross_fold_data.csv'
    if os.path.isfile(os.path.join(working_dir, cross_fold_csv_filename)):
        cross_fold_csv = pd.read_csv(os.path.join(working_dir, cross_fold_csv_filename), sep=',')
        seed_value = int((len(cross_fold_csv.columns) - 2) / 10)
        current_fold = ((len(cross_fold_csv.columns) - 2) % 10)
    else:
        seed_value = 0
        current_fold = 0
    train_ratio, val_ratio, test_ratio = 0.75, 0.15, 0.10
    train_data, val_data, test_data = split_by_patient_and_current_crossval(
        full_processed_data_address,
        train_ratio, val_ratio, test_ratio, seed_value, current_fold)

    # Compute weights for the training set
    targets = np.hstack([
        [train_data[patient][0]['score'] for patient in train_data
         if 0 in train_data[patient] and not train_data[patient][0]['failed']],
        [train_data[patient][1]['score'] for patient in train_data
         if 1 in train_data[patient] and not train_data[patient][1]['failed']],
        [train_data[patient][3]['score'] for patient in train_data
         if 3 in train_data[patient] and not train_data[patient][3]['failed']]])
    computed_weights = targets  # compute_sample_weights(targets)

    mean, std = compute_target_statistics(targets)

    row_list = []
    for patient in train_data:
        for timepoint in train_data[patient]:
            row_list.append({'case': patient, 'timepoint': timepoint,
                             'crossval{}-{}'.format(seed_value, current_fold): 'train'})
    for patient in val_data:
        for timepoint in val_data[patient]:
            row_list.append({'case': patient, 'timepoint': timepoint,
                             'crossval{}-{}'.format(seed_value, current_fold): 'val'})
    for patient in test_data:
        for timepoint in test_data[patient]:
            row_list.append({'case': patient, 'timepoint': timepoint,
                             'crossval{}-{}'.format(seed_value, current_fold): -1000})
    this_cross_val_data = pd.DataFrame(row_list, columns=['case', 'timepoint',
                                                          'crossval{}-{}'.format(seed_value, current_fold)])
    if os.path.isfile(os.path.join(working_dir, cross_fold_csv_filename)):
        cross_fold_csv = pd.read_csv(os.path.join(working_dir, cross_fold_csv_filename), sep=',')
        cross_fold_csv = pd.merge(cross_fold_csv, this_cross_val_data, on=['case', 'timepoint'])
        cross_fold_csv.to_csv(os.path.join(working_dir, cross_fold_csv_filename), index=False)
    else:
        this_cross_val_data.to_csv(os.path.join(working_dir, cross_fold_csv_filename), index=False)


    if weighted_sampling:
        sample_weights = computed_weights
    else:
        sample_weights = None

    if transformed:
        # Define your augmentations
        data_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            # Assuming images are PIL images; if not, you'll need to adjust or implement suitable transformations
        ])
    else:
        data_transforms = None

    # Create Dataset
    print("Creating Dataset", time.localtime())
    train_dataset = MediciLoader(train_data, transform=data_transforms)
    val_dataset = MediciLoader(val_data)
    test_dataset = MediciLoader(test_data)

    # Create DataLoaders
    print("Creating DataLoaders for", device, time.localtime())
    if by_patient:
        if weighted_sampling:
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=WeightedRandomSampler(weights=sample_weights,
                                                                    num_samples=len(train_dataset),
                                                                    replacement=True),
                                      collate_fn=custom_collate,
                                      generator=torch.Generator(device=device))
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate,
                                      generator=torch.Generator(device=device))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate,
                                generator=torch.Generator(device=device))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate,
                                 generator=torch.Generator(device=device))
    else:
        if weighted_sampling:
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=WeightedRandomSampler(weights=sample_weights,
                                                                    num_samples=len(train_dataset),
                                                                    replacement=True),
                                      generator=torch.Generator(device=device))
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      generator=torch.Generator(device=device))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                generator=torch.Generator(device=device))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 generator=torch.Generator(device=device))

    return train_loader, val_loader, test_loader, seed_value, current_fold

def return_combined_loaders(file_name_1, file_name_2, transformed, weighted_loss, weighted_sampling, batch_size, seed_value=0,
                       only_testing=False, only_first=False):
    print("Beginning data loading", time.localtime())

    full_processed_data_address_1 = os.path.join(processed_dataset_path, file_name_1+'.pth')
    full_processed_data_address_2 = os.path.join(processed_dataset_path, file_name_2+'.pth')

    print(f"Data being collected = {file_name_1} and {file_name_2} from {processed_dataset_path}")
    print(time.localtime())
    print("Processing data for the first time", time.localtime())
    # Splitting the dataset
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    train_data1, val_data1, test_data1 = split_and_group_by_patient(full_processed_data_address_1,
                                                       train_ratio, val_ratio, seed_value)
    train_data2, val_data2, test_data2 = split_by_patient(full_processed_data_address_2,
                                                       train_ratio, val_ratio, seed_value)
    if only_first:
        train_data = train_data1
        val_data = val_data1
        test_data = test_data1
    else:
        train_data = train_data1 + train_data2
        val_data = val_data1 + val_data2
        test_data = test_data1 + test_data2

    # Compute weights for the training set
    targets = [label for _, label, _, _, _, _ in train_data]
    computed_weights = targets  # compute_sample_weights(targets)

    mean, std = compute_target_statistics(targets)

    if weighted_sampling:
        sample_weights = computed_weights
    else:
        sample_weights = None

    if transformed:
        # Define your augmentations
        data_transforms = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            # Assuming images are PIL images; if not, you'll need to adjust or implement suitable transformations
        ])
    else:
        data_transforms = None

    # Create Dataset
    print("Creating Dataset", time.localtime())
    val_dataset = MosaicEvaluateLoader(val_data, max_n=4)
    if optuna_optimisation:
        train_dataset = MosaicDataset(train_data, transform=data_transforms, weights=sample_weights)
        test_dataset = MosaicEvaluateLoader(test_data, max_n=4)
    else:
        train_data.extend(test_data)
        train_dataset = MosaicDataset(train_data, transform=data_transforms, weights=sample_weights)
        test_dataset = MosaicEvaluateLoader(test_data[-10:])

    # Create DataLoaders
    print("Creating DataLoaders for", device, time.localtime())
    if by_patient:
        if weighted_sampling:
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=WeightedRandomSampler(weights=sample_weights,
                                                                    num_samples=len(train_dataset),
                                                                    replacement=True),
                                      collate_fn=custom_collate,
                                      generator=torch.Generator(device=device))
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate,
                                      generator=torch.Generator(device=device))
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate,
                                generator=torch.Generator(device=device))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate,
                                 generator=torch.Generator(device=device))
    else:
        if weighted_sampling:
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      sampler=WeightedRandomSampler(weights=sample_weights,
                                                                    num_samples=len(train_dataset),
                                                                    replacement=True),
                                      generator=torch.Generator(device=device))
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      generator=torch.Generator(device=device))
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                generator=torch.Generator(device=device))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 generator=torch.Generator(device=device))

    return train_loader, val_loader, test_loader