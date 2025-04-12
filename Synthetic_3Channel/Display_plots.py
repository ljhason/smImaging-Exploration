#error in code!!
from pathlib import Path
import sys

current_file_path = Path(__file__)
# Get the parent directory of the current directory (i.e., SummerProject)
parent_directory = current_file_path.parent.parent
sys.path.append(str(parent_directory))
from pma_open import *

#My image
hel1_file_path = 'Synthetic_3Channel/Synthetic Images hel1_new/Simple_3CH_hel1.pma'
avg_image_path = 'Synthetic_3Channel/Simple_3CH_hel1_Avg_Frame/Simple_3CH_hel1_Avg_Frame.png'
hel1_avg_image = io.imread(avg_image_path)

#Chanel image paths
CH1_img_path = "Synthetic_3Channel/Simple_3CH_hel1_Avg_Frame/hel1_Avg_Frame_CH1.png"
CH2_img_path = "Synthetic_3Channel/Simple_3CH_hel1_Avg_Frame/hel1_Avg_Frame_CH2.png"
CH3_img_path = "Synthetic_3Channel/Simple_3CH_hel1_Avg_Frame/hel1_Avg_Frame_CH3.png"

good_peaks_1,_ = good_peak_finder(CH1_img_path)
good_peaks_2,_ = good_peak_finder(CH2_img_path)
good_peaks_3,_ = good_peak_finder(CH3_img_path)

# Move good_peaks_1 to CH2 to display full image
good_peaks_2_CH2 = shift_peaks(good_peaks_2, shift=[0, 171])
good_peaks_3_CH3 = shift_peaks(good_peaks_3, shift=[0, 342])

#new coords for CH2:
CH1_array = np.array([[63, 15], [51, 115], [120, 50], [108, 125], [210, 51], [228, 134], [322, 42], [327, 140], [422, 45], [420, 143]])
CH2_array = np.array([[67, 186], [54, 288], [123,222], [110, 297], [212, 222], [229, 306], [323, 213], [327,311], [422, 216], [419, 313]])
CH3_array = np.array([[67, 356], [54, 458], [123, 392], [110,467], [212, 392], [229, 476], [323, 383], [327,481], [422, 386], [419, 483]])
good_peaks_2_CH2_shift = shift_peaks(good_peaks_2_CH2, shift=[-1, -10])
good_peaks_3_CH3_shift = shift_peaks(good_peaks_3_CH3, shift=[-1, -11])

params_x_CH12, params_y_CH12, params_x_CH13, params_y_CH13 = find_polyfit_params_3CH(CH1_array, CH2_array, CH3_array, degree=3)
mapped_CH2 = apply_polyfit_params(good_peaks_1, params_x_CH12, params_y_CH12).astype(np.uint16)
mapped_CH3 = apply_polyfit_params(good_peaks_1, params_x_CH13, params_y_CH13).astype(np.uint16)
count_out_pair_arr_CH1, out_pair_arr_CH1, out_pair_arr_CH2, out_pair_arr_CH3 = find_trip(good_peaks_1, mapped_CH2, mapped_CH3, tolerance=5, shift_CH2=[-1, -10], shift_CH3=[-1, -11])


y_centres = np.concatenate(( out_pair_arr_CH1[:,0], out_pair_arr_CH2[:,0], out_pair_arr_CH3[:,0]))
x_centres = np.concatenate(( out_pair_arr_CH1[:,1], out_pair_arr_CH2[:,1], out_pair_arr_CH3[:,1]))
circle_array_new = draw_circle(4, y_centres, x_centres, hel1_avg_image.shape[0])


mask = (circle_array_new == [255, 255, 0]).all(axis=-1)
if hel1_avg_image.ndim == 2:
    image_copy = hel1_avg_image.copy()
    image_copy = np.repeat(hel1_avg_image[..., np.newaxis], 3, -1)
elif hel1_avg_image.ndim==3 and hel1_avg_image.shape[2]==3:
    image_copy = hel1_avg_image.copy()

image_copy[mask] = [255, 255, 0]

fig, ax = plt.subplots(figsize=(9, 9))
ax.set_position([0.01, 0.3, 0.4, 0.6]) #[left, bottom, width, height]
ax.imshow(image_copy)
ax.grid()

scat1 = ax.scatter(out_pair_arr_CH1[:,1], out_pair_arr_CH1[:,0], s=50, facecolors='none', edgecolors='g', alpha=0)
scat2 = ax.scatter(out_pair_arr_CH2[:,1], out_pair_arr_CH2[:,0], s=50, facecolors='none', edgecolors='b', alpha=0)
scat3 = ax.scatter(out_pair_arr_CH3[:,1], out_pair_arr_CH3[:,0], s=50, facecolors='none', edgecolors='r', alpha=0)
ax.set_title("Mapped Peaks: Click to Zoom in on a Peak")

scatter_data = [(scat1, out_pair_arr_CH1, "CH1"), (scat2, out_pair_arr_CH2, "CH2"), (scat3, out_pair_arr_CH3, "CH3")]

annot = init_annot(ax=ax)

fig.canvas.mpl_connect("button_press_event", lambda event: display_three_peaks(event, fig, ax, scatter_data, image_copy, hel1_avg_image, zoom_size=6))
fig.canvas.mpl_connect("motion_notify_event", lambda event: display_three_peaks(event, fig, ax, scatter_data, image_copy, hel1_avg_image, zoom_size=6))

plt.show()