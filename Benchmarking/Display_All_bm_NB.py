#error in code!!
from pathlib import Path
import sys

current_file_path = Path(__file__)
# Get the parent directory of the current directory (i.e., SummerProject)
parent_directory = current_file_path.parent.parent
sys.path.append(str(parent_directory))
from pma_open import *

#My image
file_path = "new files _24_3_25/hel1.pma"
image_path = "Benchmarking/hel1_Avg/hel1_Avg_Frame.png"
image = io.imread(image_path, as_gray=True)

#Chanel image paths
CH1_img_path = "Benchmarking/hel1_Avg/hel1_Avg_Frame_CH1.png"
CH2_img_path = "Benchmarking/hel1_AVg/hel1_Avg_Frame_CH2.png"

good_peaks_1,_ = good_peak_finder(CH1_img_path)
good_peaks_2_new,_ = good_peak_finder(CH2_img_path)

# Move good_peaks_1 to CH2 to display full image
good_peaks_1_CH2 = shift_peaks(good_peaks_1)
good_peaks_2_CH2 = shift_peaks(good_peaks_2_new)

#new coords for CH2:
CH1_peaks_10 = np.array([[18,92], [16,213], [108,43], [106, 176], [210,51], [234, 219], [366,12], [322,192], [478, 106], [502,160]])
CH2_peaks_10_new = np.array([[22,349],[19,470],[111,300],[108,433],[212,307],[234,475],[367,268],[321,448],[476,361],[499,414]])
params_x_man_10_new, params_y_man_10_new = find_polyfit_params(CH1_peaks_10, CH2_peaks_10_new, degree=3)
mapped_peaks_10_new = apply_polyfit_params(good_peaks_1, params_x_man_10_new, params_y_man_10_new).astype(np.uint16)
poly_pair_count, poly_pair_arr_CH1, poly_pair_arr_CH2 = find_pairs(good_peaks_1, mapped_peaks_10_new, tolerance=3, Channel_count=2, shift=[-1,-10])


y_centres = np.concatenate(( poly_pair_arr_CH1[:,0], poly_pair_arr_CH2[:,0]))
x_centres = np.concatenate(( poly_pair_arr_CH1[:,1], poly_pair_arr_CH2[:,1]))
circle_array_new = draw_circle(4, y_centres, x_centres, image.shape[0])


mask = (circle_array_new == [255, 255, 0]).all(axis=-1)
if image.ndim == 2:
    image_copy = image.copy()
    image_copy = np.repeat(image[..., np.newaxis], 3, -1)
elif image.ndim==3 and image.shape[2]==3:
    image_copy = image.copy()

image_copy[mask] = [255, 255, 0]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.set_position([0.01, 0.3, 0.4, 0.6]) #[left, bottom, width, height]
ax.imshow(image_copy)
ax.grid()

scat1 = ax.scatter(poly_pair_arr_CH1[:,1], poly_pair_arr_CH1[:,0], s=50, facecolors='none', edgecolors='g', alpha=0)
scat2 = ax.scatter(poly_pair_arr_CH2[:,1], poly_pair_arr_CH2[:,0], s=50, facecolors='none', edgecolors='b', alpha=0)
ax.set_title(f"Mapped Peaks ({poly_pair_count} Pairs): Click For Plots")

scatter_data = [(scat1, poly_pair_arr_CH1 , "CH1"), (scat2, poly_pair_arr_CH2, "CH2")]

annot = init_annot(ax=ax)

fig.canvas.mpl_connect("button_press_event", lambda event: on_hover_intensity(event, file_path, fig, ax, scatter_data, y_centres, x_centres, image_copy, image, mask = (circle_array_new == [255, 255, 0]).all(axis=-1), tpf=1/5, time_interval=10, background_treatment = "None", CH_consideration=True))
fig.canvas.mpl_connect("motion_notify_event", lambda event: on_hover_intensity(event, file_path, fig, ax, scatter_data, y_centres, x_centres, image_copy, image, mask = (circle_array_new == [255, 255, 0]).all(axis=-1), tpf=1/5, time_interval=10, background_treatment = "None", CH_consideration=True))

plt.show()