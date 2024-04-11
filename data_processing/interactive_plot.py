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
csv_name = 'volpara_priors_testing_CC.csv'
# csv_name = 'average_mosaic_performance.csv'
# csv_name = 'individual_mosaic_performance.csv'
df = pd.read_csv(os.path.join(csv_directory, csv_name), sep=',')
image_dir_path = 'Z:\\PROCAS_ALL_PROCESSED\\'

valid_data = pd.DataFrame()

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

    # Create a new Seaborn JointPlot
    jointplot = sns.jointplot(x=df[selected_x], y=df[selected_y], kind="scatter")

    # Add the MSE and R2 to the plot title
    ax_joint = jointplot.ax_joint

    # Add a faint dashed x=y line
    ax_joint.plot([df[selected_x].min(), df[selected_x].max()],
                  [df[selected_y].min(), df[selected_y].max()],
                  linestyle='--', color='red')

    # Add the MSE and R2 as a text box to the right of the plot
    text = f'MSE: {mse:.2f}\nR²: {r2:.2f}'
    plt.figtext(0.93, 0.5, text, ha='center', va='center', fontsize=12)

    # Get the current size of the canvas in pixels
    width, height = canvas_widget.winfo_width(), canvas_widget.winfo_height()

    # Convert canvas size from pixels to inches for Matplotlib
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

    
def getPointsNearest(valid_data, xdata, ydata, selected_x, selected_y):
    points = pd.DataFrame()
    count = 1
    while points.empty:
        points = valid_data.loc[((df[selected_x] >= xdata - count) & (df[selected_x] <= xdata + count)) & ((df[selected_y] >= ydata - count) & (df[selected_y] <= ydata + count))]
        count = count * 1.5
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
    
    foundPoints = getPointsNearest(valid_data, event.xdata, event.ydata, selected_x, selected_y)
    print(foundPoints)
    fullFoundPoints = df.iloc[foundPoints.index.values]
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
        img_paths = sorted(os.listdir(patient_path))

        print(img_paths)
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(patient_path)        

        ds = dicom.dcmread(patient_path + '/' + img_paths[0])
        img = ds.pixel_array.astype(float) 
        axs[0, 1].imshow(img, cmap='gray')
        axs[0, 1].set_title('LCC')

        ds = dicom.dcmread(patient_path + '/' + img_paths[1])
        img = ds.pixel_array.astype(float) 
        axs[1, 1].imshow(img, cmap='gray')
        axs[1, 1].set_title('LMLO')

        ds = dicom.dcmread(patient_path + '/' + img_paths[2])
        img = ds.pixel_array.astype(float) 
        axs[0, 0].imshow(img, cmap='gray')
        axs[0, 0].set_title('RCC')
        
        ds = dicom.dcmread(patient_path + '/' + img_paths[3])
        img = ds.pixel_array.astype(float) 
        axs[1, 0].imshow(img, cmap='gray')
        axs[1, 0].set_title('RMLO')

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
