from pathlib import Path
import sys

current_file_path = Path(__file__)
# Get the parent directory of the current directory (i.e., SummerProject)
parent_directory = current_file_path.parent.parent
sys.path.append(str(parent_directory))
from pma_open import *


from matplotlib.widgets import Cursor, Button

#My image
file_path = 'Dropbox Files/hel1.pma'
image_path = "img_avg_map/hel1_Avg_Frame/hel1_Avg_Frame.png"
image = io.imread(image_path, as_gray=True)

#Image on dropbox
hel1_ave_path = 'Dropbox Files/hel1_ave_LF_Circ.tif'
hel1_ave_image = io.imread(hel1_ave_path)

#Testing avg_frame_arr function
avg_frame_data = avg_frame_arr(file_path)

#Chanel arrays
avg_frame_data_CH1 = avg_frame_data[:,:256]
avg_frame_data_CH2 = avg_frame_data[:,256:]

#Chanel image paths
CH1_img_path = "img_avg_map/hel1_Avg_Frame/hel1_Avg_Frame_CH1.png"
CH2_img_path = "img_avg_map/hel1_Avg_Frame/hel1_Avg_Frame_CH2.png"

#Chanel images
image_CH1 = io.imread(CH1_img_path, as_gray=True)
image_CH2 = io.imread(CH2_img_path, as_gray=True)

peaks_coords_IDL_CH1 = find_peaks_scipy_IDL(CH1_img_path)[0]
peaks_coords_IDL_CH2_new = find_peaks_scipy_IDL(CH2_img_path, sigma=2, block_size=16, scaler_percent=10)[0]

good_peaks_1,_ = good_peak_finder_CH1(CH1_img_path)
good_peaks_2_new,_ = good_peak_finder_CH2(CH2_img_path, sigma=2, block_size=16, scaler_percent=10, boarder=10, max_rad=3)

# Move good_peaks_1 to CH2 to display full image
good_peaks_1_CH2 = shift_peaks_CH(good_peaks_1)
good_peaks_2_CH2 = shift_peaks_CH(good_peaks_2_new)

#Changing any parameters to see if the blue circle more closely resemble the red circles. 
fig, ax = plt.subplots(figsize=(4, 8))

ax.set_title("CH2 Image Peak Comparision")
# ax.imshow(image_CH2, cmap="gray", alpha=0.9)
scat1 = ax.scatter(good_peaks_1[:, 1], good_peaks_1[:, 0], s=50, facecolors='none', edgecolors='r', label='Peaks from CH1')
scat2 = ax.scatter(good_peaks_2_new[:, 1], good_peaks_2_new[:, 0], s=50, facecolors='none', edgecolors='b', label='Peaks from CH2')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Create annotation text box
annot = ax.annotate("", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

# Function to update annotation text and position
def update_annot(ind, scatter, peaks, label):
    """ Updates the annotation position and text """
    idx = ind["ind"][0]
    x, y = peaks[idx]
    annot.xy = (scatter.get_offsets()[idx][0], scatter.get_offsets()[idx][1])
    annot.set_text(f"{label} Peak {idx}: ({x}, {y})")
    annot.set_visible(True)

# Event listener for hover functionality
def on_hover(event):
    """ Checks if the mouse hovers over a point and updates annotation """
    visible = False
    for scatter, peaks, label in [(scat1, good_peaks_1, "CH1"), (scat2, good_peaks_2_CH2, "CH2")]:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, scatter, peaks, label)
            visible = True
            break
    annot.set_visible(visible)
    fig.canvas.draw_idle()

def on_click(event):
    visible = False
    for scatter, peaks, label in [(scat1, good_peaks_1, "CH1"), (scat2, good_peaks_2_CH2, "CH2")]:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, scatter, peaks, label)
            visible = True
            break
    annot.set_visible(visible)
    fig.canvas.draw_idle()
    #print Chanel, peak index and peak coordinates
    print(f"{label} Peak {ind['ind'][0]}: ({peaks[ind['ind'][0]][0]}, {peaks[ind['ind'][0]][1]})")


# Connect hover event to the figure
fig.canvas.mpl_connect("motion_notify_event", on_hover)
fig.canvas.mpl_connect("button_press_event", on_click)

plt.show()


