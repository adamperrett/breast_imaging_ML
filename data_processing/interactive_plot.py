import os
import tkinter as tk
from tkinter import ttk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error, r2_score
import glob
import pydicom as dicom

# Read the CSV file
csv_directory = 'C:/Users/adam_/PycharmProjects/breast_imaging_ML/csv_data'
# csv_name = 'priors_for_all.csv'
# csv_name = 'updated_all_priors.csv'
# csv_name = 'priors_per_image_reader_and_MAI.csv'
# csv_name = 'PROCAS_Volpara_dirty.csv'
# csv_name = 'volpara_priors_testing.csv'
csv_name = 'volpara_priors_testing_weight.csv'
# csv_name = 'average_mosaic_performance.csv'
# csv_name = 'individual_mosaic_performance.csv'
df = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')
image_dir_path = 'Z:\\PROCAS_ALL_PROCESSED\\'

truth_labels = 'priors'
if truth_labels == 'priors':
    vas_csv_location = 'priors_per_image_reader_and_MAI.csv'
    vas_df = pd.read_csv(os.path.join(csv_directory, vas_csv_location), sep=',')

valid_data = pd.DataFrame()

# add something to plot it as error vs actual + stdev bars instead
plotting_type = 'birads'
birad_intervals = [3.5, 7.5, 15.5]
birad_categories = [1, 2, 3, 4]

columns_to_convert = ['ave_MAI_VAS_PRO', 'ave_MAI_VAS_RAW']
# Convert each column to numeric
for col in columns_to_convert:
    if col in df:
        df[col] = pd.to_numeric(df[col], errors='coerce')

def resize_plot(event=None):
    # Delay resizing to allow window and canvas to update dimensions
    window.after(100, perform_resizing)

def perform_resizing():
    dpi = fig.get_dpi()
    width, height = canvas_widget.winfo_width(), canvas_widget.winfo_height()
    # Check if dimensions are valid
    if width > 1 and height > 1:
        fig.set_size_inches(width / dpi, height / dpi)
        canvas.draw_idle()

def update_plot():
    selected_x = x_variable.get()
    selected_y = y_variable.get()

    # Filter out NaN values for accurate calculations
    valid_data = df[[selected_x, selected_y]].dropna()

    # Calculate MSE and R2
    mse = mean_squared_error(valid_data[selected_x], valid_data[selected_y])
    r2 = r2_score(valid_data[selected_x], valid_data[selected_y])
    wx_r2 = r2_score(valid_data[selected_x], valid_data[selected_y], sample_weight=valid_data[selected_x])
    wy_r2 = r2_score(valid_data[selected_x], valid_data[selected_y], sample_weight=valid_data[selected_y])

    error = df[selected_y] - df[selected_x]
    import numpy as np
    stderr = np.std(error)
    aveerr = np.mean(error)
    abserr = np.mean(np.abs(error))
    # Create a new Seaborn JointPlot
    if plotting_type == 'error':
        interval = stderr * 1.95
        jointplot = sns.jointplot(x=df[selected_x], y=error, kind="scatter")
        # Add the MSE and R2 to the plot title
        ax_joint = jointplot.ax_joint

        # Add a faint dashed y=0 line
        min_v = min(df[selected_x].min(), df[selected_y].min())
        max_v = max(df[selected_x].max(), df[selected_y].max())
        ax_joint.plot([min_v, max_v],
                      [0, 0],
                      linestyle='-', color='black')
        ax_joint.plot([min_v, max_v],
                      [aveerr, aveerr],
                      linestyle='--', color='red')
        ax_joint.plot([min_v, max_v],
                      [interval, interval],
                      linestyle='--', color='blue')
        ax_joint.plot([min_v, max_v],
                      [-interval, -interval],
                      linestyle='--', color='blue')
        ax_joint.set_ylabel('Error')
    elif plotting_type == 'birads':
        if 'VBD' not in selected_x or 'VBD' not in selected_y:
            print("The BIRADS plotting is setup to expect columns beginning VBD, the average. Please try again.")
        else:
            columns_to_max_x = [selected_x.replace('VBD', suffix) for suffix in
                                ['vbd_L_CC', 'vbd_R_CC', 'vbd_L_MLO', 'vbd_R_MLO']]
            columns_to_max_y = [selected_y.replace('VBD', suffix) for suffix in
                                ['vbd_L_CC', 'vbd_R_CC', 'vbd_L_MLO', 'vbd_R_MLO']]

            df['max_x'] = df[columns_to_max_x].dropna().max(axis=1)
            df['max_y'] = df[columns_to_max_y].dropna().max(axis=1)

            df['cat_x'] = pd.cut(df['max_x'], bins=[-np.inf] + birad_intervals + [np.inf], labels=birad_categories)
            df['cat_y'] = pd.cut(df['max_y'], bins=[-np.inf] + birad_intervals + [np.inf], labels=birad_categories)

            fig_sp, axes = plt.subplots(4, 4, figsize=(16, 16))

            # Iterate over each category combination and plot
            for i, cat_x in enumerate(birad_categories):
                for j, cat_y in enumerate(birad_categories):
                    ax = axes[i, j]
                    # Filter data for the current category combination
                    subset = df[(df['cat_x'] == cat_x) & (df['cat_y'] == cat_y)]
                    if not len(subset):
                        continue

                    # Scatter plot for the current categories
                    sns.scatterplot(x='max_x', y='max_y', data=subset, ax=ax, color='blue', alpha=0.6)

                    # Add x=y line
                    min_val = min(subset['max_x'].min(), subset['max_y'].min())
                    max_val = max(subset['max_x'].max(), subset['max_y'].max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--')

                    # Set plot limits
                    ax.set_xlim(min_val, max_val)
                    ax.set_ylim(min_val, max_val)

                    # Optional: Set titles or labels if desired
                    ax.set_title(f'{len(subset)} in BIRADS X: {cat_x}, BIRADS Y: {cat_y}')
                    # ax.set_xlabel('Max X')
                    # ax.set_ylabel('Max Y')

    else:
        jointplot = sns.jointplot(x=df[selected_x], y=df[selected_y], kind="scatter")
        # Add the MSE and R2 to the plot title
        ax_joint = jointplot.ax_joint

        # Add a faint dashed x=y line
        min_v = min(df[selected_x].min(), df[selected_y].min())
        max_v = max(df[selected_x].max(), df[selected_y].max())
        ax_joint.plot([min_v, max_v],
                      [min_v, max_v],
                      linestyle='--', color='red')

    # Add the MSE and R2 as a text box to the right of the plot
    text = f'MSE: {mse:.2f}\nR²: {r2:.2f}\nx_R²: {wx_r2:.2f}\ny_R²: {wy_r2:.2f}\n√err²: {abserr:.2f}\nstd: {stderr:.2f}'
    plt.figtext(0.93, 0.5, text, ha='center', va='center', fontsize=12)

    # Get the current size of the canvas in pixels
    width, height = canvas_widget.winfo_width(), canvas_widget.winfo_height()

    if plotting_type == 'birads':
        dpi = fig_sp.get_dpi()
        fig_sp.set_size_inches(width / dpi, height / dpi)
        fig_sp.set_dpi(dpi)

        # Draw and update the canvas
        canvas.figure = fig_sp
        canvas.draw_idle()

        # Optional: connect button press event
        fig_sp.canvas.mpl_connect('button_press_event', onclick)
    else:
        dpi = fig.get_dpi()
        fig_width = width / dpi
        fig_height = height / dpi

        # Set the size of the new figure to match the canvas size
        jointplot.fig.set_size_inches(fig_width, fig_height)
        jointplot.fig.set_dpi(dpi)

        jointplot.figure.canvas.mpl_connect('button_press_event', onclick)

        # Update the Tkinter canvas with the new figure
        canvas.figure = jointplot.fig
        canvas.draw_idle()

    
def getPointsNearest(valid_data, xdata, ydata, x_values, y_values):
    points = pd.DataFrame()
    count = 0.01
    while points.empty:
        points = valid_data.loc[((x_values >= xdata - count) & (x_values <= xdata + count)) & ((y_values >= ydata - count) & (y_values <= ydata + count))]
        count = count * 1.01
    return points

def onclick(event):
    print("Pressed")
    if not (hasattr(event, 'xdata') & hasattr(event, 'ydata')):
        return
    
    selected_x = x_variable.get()
    selected_y = y_variable.get()
    print(selected_x, selected_y)
    
    if (selected_x == '') & (selected_y == ''):
        return
    
    valid_data = df[[selected_x, selected_y]].dropna()

    if plotting_type == 'error':
        x_values = df[selected_x]
        y_values = df[selected_x] - df[selected_y]
    elif plotting_type == 'birads':
        x_values = df['max_x']
        y_values = df['max_y']
    else:
        x_values = df[selected_x]
        y_values = df[selected_y]

    foundPoints = getPointsNearest(valid_data, event.xdata, event.ydata, x_values, y_values)

    print(foundPoints)
    fullFoundPoints = df.iloc[foundPoints.index.values].drop_duplicates(subset=['ASSURE_PROCESSED_ANON_ID'])
    if fullFoundPoints.shape[0] <= 2:
        showDicomsInDf(fullFoundPoints)
    else:
        print(f'{fullFoundPoints.shape[0]} point(s) found')
        print(fullFoundPoints)


def showDicomsInDf(foundPoints):
    print(image_dir_path + '*' + "{:05d}".format(int(foundPoints.ASSURE_PROCESSED_ANON_ID.values[0])) + '*')
    patient_paths = glob.glob(image_dir_path + '*' + "{:05d}".format(int(foundPoints.ASSURE_PROCESSED_ANON_ID.values[0])) + '*')
    if(len(patient_paths) == 0):
        print('None found')
    for patient_path in patient_paths:
        patient_data = vas_df.loc[vas_df['ASSURE_PROCESSED_ANON_ID'] == int(foundPoints.ASSURE_PROCESSED_ANON_ID.values[0])]

        img_paths = sorted(os.listdir(patient_path))

        print(img_paths)
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("VAS: {} for {}".format(patient_data['VASCombinedAvDensity'].values[0], patient_path))

        ds = dicom.dcmread(patient_path + '/' + img_paths[0])
        img = ds.pixel_array.astype(float) 
        axs[0, 1].imshow(img, cmap='gray')
        LCC_1 = patient_data['LCC-1'].values[0]
        LCC_2 = patient_data['LCC-2'].values[0]
        LCC_ave = patient_data['LCC'].values[0]
        axs[0, 1].set_title('LCC - r₁:{} - r₂:{} - ave:{}'.format(LCC_1, LCC_2, LCC_ave))

        ds = dicom.dcmread(patient_path + '/' + img_paths[1])
        img = ds.pixel_array.astype(float) 
        axs[1, 1].imshow(img, cmap='gray')
        LMLO_1 = patient_data['LMLO-1'].values[0]
        LMLO_2 = patient_data['LMLO-2'].values[0]
        LMLO_ave = patient_data['LMLO'].values[0]
        axs[1, 1].set_title('LMLO - r₁:{} - r₂:{} - ave:{}'.format(LMLO_1, LMLO_2, LMLO_ave))

        ds = dicom.dcmread(patient_path + '/' + img_paths[2])
        img = ds.pixel_array.astype(float) 
        axs[0, 0].imshow(img, cmap='gray')
        RCC_1 = patient_data['RCC-1'].values[0]
        RCC_2 = patient_data['RCC-2'].values[0]
        RCC_ave = patient_data['RCC'].values[0]
        axs[0, 0].set_title('RCC - r₁:{} - r₂:{} - ave:{}'.format(RCC_1, RCC_2, RCC_ave))
        
        ds = dicom.dcmread(patient_path + '/' + img_paths[3])
        img = ds.pixel_array.astype(float) 
        axs[1, 0].imshow(img, cmap='gray')
        RMLO_1 = patient_data['RMLO-1'].values[0]
        RMLO_2 = patient_data['RMLO-2'].values[0]
        RMLO_ave = patient_data['RMLO'].values[0]
        axs[1, 0].set_title('RMLO - r₁:{} - r₂:{} - ave:{}'.format(RMLO_1, RMLO_2, RMLO_ave))

        for ax in fig.get_axes():
            ax.label_outer()
            ax.axis('off')
        fig.show()
    update_plot()

# Setup window
window = tk.Tk()
window.title("Interactive Jointplot")

window.grid_rowconfigure(1, weight=1)
window.grid_columnconfigure(0, weight=1)

def show_dropdown(event):
    """ Function to display the dropdown list on mouse click. """
    event.widget.event_generate('<Down>')

# Create dropdown menus for column selection
x_variable = tk.StringVar(window)
y_variable = tk.StringVar(window)

# Set the width of the dropdown menus
dropdown_width = 50  # Adjust this value as needed

x_dropdown = ttk.Combobox(window, textvariable=x_variable, values=df.columns.tolist(), width=dropdown_width)
y_dropdown = ttk.Combobox(window, textvariable=y_variable, values=df.columns.tolist(), width=dropdown_width)

# Bind the dropdowns to show list on click
x_dropdown.bind("<Button-1>", show_dropdown)
y_dropdown.bind("<Button-1>", show_dropdown)

# Grid the dropdown menus
x_dropdown.grid(column=0, row=0)
y_dropdown.grid(column=1, row=0)

# Button to update the plot
update_button = tk.Button(window, text="Update Plot", command=update_plot)
update_button.grid(column=2, row=0)

# Matplotlib Figure and a canvas widget
fig = plt.Figure()
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.mpl_connect('key_press_event', onclick)
canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(column=0, row=1, columnspan=3, sticky='nsew')

# Bind the resize event
window.bind("<Configure>", resize_plot)

window.mainloop()

print("Done")
