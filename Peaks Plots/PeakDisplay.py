from pathlib import Path
import sys

current_file_path = Path(__file__)
# Get the parent directory of the current directory (i.e., SummerProject)
parent_directory = current_file_path.parent.parent
sys.path.append(str(parent_directory))
from pma_open import *

#My image
file_path = 'Dropbox Files/hel1.pma'
image_path = "Channel Mapping/hel1_Avg_Frame/hel1_Avg_Frame.png"
image = io.imread(image_path, as_gray=True)

#Chanel image paths
CH1_img_path = "Channel Mapping/hel1_Avg_Frame/hel1_Avg_Frame_CH1.png"
CH2_img_path = "Channel Mapping/hel1_Avg_Frame/hel1_Avg_Frame_CH2.png"

good_peaks_1,_ = good_peak_finder(CH1_img_path)
good_peaks_2_new,_ = good_peak_finder(CH2_img_path, sigma=2, block_size=16, scaler_percent=10, boarder=10, max_rad=3)

# Move good_peaks_1 to CH2 to display full image
good_peaks_1_CH2 = shift_peaks(good_peaks_1)
good_peaks_2_CH2 = shift_peaks(good_peaks_2_new)

# Poly Mapping
CH1_peaks_10 = np.array([[55,63], [14, 194], [179, 45], [115, 172], [273, 65], [257, 192], [358, 91], [365, 199], [419, 44], [445, 205]])
CH2_peaks_10 = np.array([[60, 322], [18, 453], [183, 304], [119, 431], [276, 323], [258, 450], [359, 350], [364, 456], [420, 301], [446, 462]])

params_x_man_10, params_y_man_10 = find_polyfit_params(CH1_peaks_10, CH2_peaks_10, degree=3)
mapped_peaks_10 = apply_polyfit_params(good_peaks_1, params_x_man_10, params_y_man_10).astype(np.uint16)
poly_pair_count_tol4_10, poly_pair_arr_CH1_tol4_10, poly_pair_arr_CH2_tol4_10 = find_polyfit_pairs(mapped_peaks_10, good_peaks_1, tolerance=4)

#Interactive Plots
fig = plt.figure(figsize=(8, 8))
ax = fig.subplots()

ax.imshow(image, alpha=0.8, cmap="gray")
ax.set_title(f"Polymap, tolerance=4 ({poly_pair_count_tol4_10} pairs)")
scat1=ax.scatter(poly_pair_arr_CH1_tol4_10[:, 1], poly_pair_arr_CH1_tol4_10[:, 0], s=50, facecolors='none', edgecolors='y', label='Poly Pairs')
scat2=ax.scatter(poly_pair_arr_CH2_tol4_10[:, 1], poly_pair_arr_CH2_tol4_10[:, 0], s=50, facecolors='none', edgecolors='y')
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.06))

annot = init_annot(ax=ax)

scatter_data = [(scat1, poly_pair_arr_CH1_tol4_10, "CH1"), (scat2, poly_pair_arr_CH2_tol4_10, "CH2")]
# Connect hover event to the figure
fig.canvas.mpl_connect("motion_notify_event", lambda event: print_coords_trigger(event, fig, scatter_data))
fig.canvas.mpl_connect("button_press_event", lambda event: print_coords_trigger(event, fig, scatter_data))

plt.show()
