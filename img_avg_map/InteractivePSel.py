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
good_peaks_1_CH2 = shift_peaks(good_peaks_1)
good_peaks_2_CH2 = shift_peaks(good_peaks_2_new)


fig = plt.figure(figsize=(8, 8))
ax = fig.subplots()
plt.axhline(y= 102, color='w', linestyle='--')  
plt.axhline(y= 204, color='w', linestyle='--')
plt.axhline(y= 308, color='w', linestyle='--')
plt.axhline(y= 410, color='w', linestyle='--')

plt.axvline(x= 128, color='w', linestyle='--')
plt.axvline(x= 384, color='w', linestyle='--')

plt.axvline(x= 256, color='w', linestyle='-')

plt.suptitle("CH1 and CH2 Identified Peaks", fontsize=16)
plt.title("Hover over points to see peak index and coordinates \n Click on peaks to print peak info in terminal \n Indentify and select corresponding peaks in CH1 and CH2 from each third of the image", fontsize=10)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.imshow(image, cmap="gray", alpha=0.7)
scat1 = ax.scatter(good_peaks_1[:, 1], good_peaks_1[:, 0], s=50, facecolors='none', edgecolors='r', label='Peaks from CH1')
scat2 = ax.scatter(good_peaks_2_CH2[:, 1], good_peaks_2_CH2[:, 0], s=50, facecolors='none', edgecolors='b', label='Peaks from CH2')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15))

annot = init_annot(ax=ax)

scatter_data = [(scat1, good_peaks_1, "CH1"), (scat2, good_peaks_2_CH2, "CH2")]
# Connect hover event to the figure
fig.canvas.mpl_connect("motion_notify_event", lambda event: on_event(event, fig, scatter_data))
fig.canvas.mpl_connect("button_press_event", lambda event: on_event(event, fig, scatter_data))

plt.show()



