import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = "C:/Users/adam_/OneDrive/Documents/Research/Breast cancer/Medici/recurrence results/focal_sample_bug_fix"

VAL_TAG = "Best AUC/Validation AUC from AUC"
TEST_TAG = "Best AUC/Test AUC from AUC"
REC_TAGS = ['breastrec', 'distrfievent', 'local', 'new_cancer']

def extract_scalar_data(event_file):
    print(f"  Extracting from: {event_file}")
    ea = EventAccumulator(event_file)
    try:
        ea.Reload()
    except Exception as e:
        print(f"  Failed to load {event_file}: {e}")
        return [], {}

    if VAL_TAG not in ea.Tags()['scalars'] or TEST_TAG not in ea.Tags()['scalars']:
        print(f"  Skipping {event_file}: required tags not found")
        return [], {}

    val_events = ea.Scalars(VAL_TAG)
    test_events = ea.Scalars(TEST_TAG)

    val_dict = {e.step: e.value for e in val_events}
    test_dict = {e.step: e.value for e in test_events}

    matched_main = [(e, val_dict[e], test_dict[e]) for e in sorted(set(val_dict) & set(test_dict))]

    # Recurrence-specific AUCs
    rec_data = {tag: [] for tag in REC_TAGS}
    for tag in REC_TAGS:
        val_tag = f"Best rec AUC/Validation {tag}"
        test_tag = f"Best rec AUC/Test {tag}"

        if val_tag in ea.Tags()['scalars'] and test_tag in ea.Tags()['scalars']:
            val_events = ea.Scalars(val_tag)
            test_events = ea.Scalars(test_tag)
            val_rec = {e.step: e.value for e in val_events}
            test_rec = {e.step: e.value for e in test_events}
            for step in sorted(set(val_rec) & set(test_rec)):
                rec_data[tag].append((step, val_rec[step], test_rec[step]))

    return matched_main, rec_data

def find_event_files(root_dir):
    print("🔍 Scanning for event files...")
    event_files = []
    for root, dirs, files in os.walk(root_dir):
        for d in dirs:
            for _, _, f in os.walk(os.path.join(root, d)):
                for setting in f:
                    if setting.startswith("events.out.tfevents"):
                        event_files.append(os.path.join(root, d, setting))
    print(f"🗂️  Found {len(event_files)} event files.")
    return event_files

def plot_xy_fit(val, test, title, ax):
    epochs = list(range(len(val)))
    norm = plt.Normalize(min(epochs), max(epochs))
    colors = cm.viridis(norm(epochs))

    ax.scatter(val, test, c=epochs, cmap='viridis', alpha=0.7)

    # x = y line
    min_val = min(min(val), min(test))
    max_val = max(max(val), max(test))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='x = y')

    # best fit line
    coeffs = np.polyfit(val, test, deg=1)
    fit_fn = np.poly1d(coeffs)
    x_vals = np.linspace(min_val, max_val, 100)
    ax.plot(x_vals, fit_fn(x_vals), 'r-', label='Best fit line')

    ax.set_title(title)
    ax.set_xlabel("Validation AUC")
    ax.set_ylabel("Test AUC")
    ax.grid(True)
    ax.legend()

def main():
    all_points = []
    rec_points = {tag: [] for tag in REC_TAGS}
    event_files = find_event_files(LOG_DIR)

    if not event_files:
        print("❌ No event files found.")
        return

    print("📥 Extracting scalar data from each event file...")
    for event_file in tqdm(event_files, desc="Processing files"):
        main_data, rec_data = extract_scalar_data(event_file)
        all_points.extend(main_data)
        for tag in REC_TAGS:
            rec_points[tag].extend(rec_data.get(tag, []))

    if not all_points:
        print("⚠️ No matching scalar data found across all runs.")
        return

    # Main validation vs test plot
    print("📊 Plotting overall validation vs. test AUC...")
    epochs, val_aucs, test_aucs = zip(*all_points)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(val_aucs, test_aucs, c=epochs, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Epoch")

    min_val = min(min(val_aucs), min(test_aucs))
    max_val = max(max(val_aucs), max(test_aucs))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='x = y')

    # Add best fit line
    coeffs = np.polyfit(val_aucs, test_aucs, deg=1)
    fit_fn = np.poly1d(coeffs)
    x_vals = np.linspace(min_val, max_val, 100)
    plt.plot(x_vals, fit_fn(x_vals), 'r-', label='Best fit line')

    plt.xlabel("Validation AUC")
    plt.ylabel("Test AUC")
    plt.title("Validation vs. Test AUC over Epochs (colored by epoch)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Subplots per recurrence tag
    print("📊 Plotting recurrence-specific subplots...")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()
    for i, tag in enumerate(REC_TAGS):
        points = rec_points[tag]
        if not points:
            axs[i].set_title(f"{tag} (no data)")
            axs[i].axis('off')
            continue
        _, val, test = zip(*points)
        plot_xy_fit(val, test, f"{tag}", axs[i])

    plt.suptitle("Recurrence Type: Validation vs Test AUC", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("✅ Done.")

if __name__ == "__main__":
    main()
