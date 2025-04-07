from pathlib import Path
import sys

current_file_path = Path(__file__)
# Get the parent directory of the current directory (i.e., SummerProject)
parent_directory = current_file_path.parent.parent
sys.path.append(str(parent_directory))
from pma_open import *

#My image
image_path = "Synthetic_3Channel/Simple_3CH_hel1_Avg_Frame/Simple_3CH_hel1_Avg_Frame.png"
image = io.imread(image_path, as_gray=True)

#Chanel image paths
CH1_img_path = "Synthetic_3Channel/Simple_3CH_hel1_Avg_Frame/hel1_Avg_Frame_CH1.png"
CH2_img_path = "Synthetic_3Channel/Simple_3CH_hel1_Avg_Frame/hel1_Avg_Frame_CH2.png"
CH3_img_path = "Synthetic_3Channel/Simple_3CH_hel1_Avg_Frame/hel1_Avg_Frame_CH3.png"
good_peaks_1,_ = good_peak_finder(CH1_img_path)
good_peaks_2,_ = good_peak_finder(CH2_img_path)
good_peaks_3,_ = good_peak_finder(CH3_img_path)

# Move good_peaks_1 to CH2 to display full image
good_peaks_2_CH2 = shift_peaks(good_peaks_2, [0,171])
good_peaks_3_CH3 = shift_peaks(good_peaks_3, [0,342])
#Linear shift to find most pairs
good_peaks_2_CH2_shift = shift_peaks(good_peaks_2_CH2, shift=[-1, -10])
good_peaks_3_CH3_shift = shift_peaks(good_peaks_3_CH3, shift=[-1, -11])

fig = plt.figure(figsize=(8, 8))
ax = fig.subplots()
plt.axhline(y= 102, color='w', linestyle='--')  
plt.axhline(y= 204, color='w', linestyle='--')
plt.axhline(y= 308, color='w', linestyle='--')
plt.axhline(y= 410, color='w', linestyle='--')

plt.axvline(x= 86, color='w', linestyle='--')
plt.axvline(x= 256, color='w', linestyle='--')
plt.axvline(x= 428, color='w', linestyle='--')

plt.axvline(x= 171, color='w', linestyle='-')
plt.axvline(x= 342, color='w', linestyle='-')

plt.suptitle("CH1, CH2 and CH3 Identified Peaks", fontsize=16)
plt.title("Hover over points to see peak index and coordinates \n Click on peaks to print peak info in terminal \n Indentify and select corresponding peaks in CH1, CH2 and CH3 from each section of the image", fontsize=10)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.imshow(image, cmap="gray")
scat1 = ax.scatter(good_peaks_1[:, 1], good_peaks_1[:, 0], s=50, facecolors='none', edgecolors='g', label='Peaks from CH1')
scat2 = ax.scatter(good_peaks_2_CH2_shift[:, 1], good_peaks_2_CH2_shift[:, 0], s=50, facecolors='none', edgecolors='orange', label='Peaks from CH2')
scat3 = ax.scatter(good_peaks_3_CH3_shift[:, 1], good_peaks_3_CH3_shift[:, 0], s=50, facecolors='none', edgecolors='pink', label='Peaks from CH3')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.15))

annot = init_annot(ax=ax)

scatter_data = [(scat1, good_peaks_1, "CH1"), (scat2, good_peaks_2_CH2_shift, "CH2"), (scat3, good_peaks_3_CH3_shift, "CH3")]
# Connect hover event to the figure
fig.canvas.mpl_connect("motion_notify_event", lambda event: print_coords_trigger(event, fig, scatter_data))
fig.canvas.mpl_connect("button_press_event", lambda event: print_coords_trigger(event, fig, scatter_data))

plt.show()



