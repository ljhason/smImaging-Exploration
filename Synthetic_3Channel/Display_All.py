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

def on_hover_display_three(event, fig, ax, scatter_data, image_3d, image_orig, zoom_size=6, CH1_zoom_axes=[0.2, 0.05, 0.15, 0.15], CH2_zoom_axes=[0.4, 0.05, 0.15, 0.15], CH3_zoom_axes=[0.6, 0.05, 0.15, 0.15]):
    """ Checks if the mouse hovers over a point and updates annotation """
    visible = False
    for scatter, peaks, label in scatter_data:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, scatter, peaks, label)
            visible = True
            idx = ind['ind'][0]

            if event.name == "button_press_event":
                for patch in ax.patches:
                    patch.remove()
                    
                for ax_zoom in fig.axes:
                    if ax_zoom is not ax:  # Keep the main axis
                        fig.delaxes(ax_zoom)

                ax_zoom_CH1 = fig.add_axes(CH1_zoom_axes)
                ax_zoom_CH2 = fig.add_axes(CH2_zoom_axes)
                ax_zoom_CH3 = fig.add_axes(CH3_zoom_axes)

                y_CH1, x_CH1 = scatter_data[0][1][idx]
                x1_CH1, x2_CH1 = max(0, x_CH1 - zoom_size), min(image_3d.shape[1], x_CH1 + zoom_size+1)
                y1_CH1, y2_CH1 = max(0, y_CH1 - zoom_size), min(image_3d.shape[0], y_CH1 + zoom_size+1)
                y_CH2, x_CH2 = scatter_data[1][1][idx]
                x1_CH2, x2_CH2 = max(0, x_CH2 - zoom_size), min(image_3d.shape[1], x_CH2 + zoom_size+1)
                y1_CH2, y2_CH2 = max(0, y_CH2 - zoom_size), min(image_3d.shape[0], y_CH2 + zoom_size+1)
                y_CH3, x_CH3 = scatter_data[2][1][idx]
                x1_CH3, x2_CH3 = max(0, x_CH3 - zoom_size), min(image_3d.shape[1], x_CH3 + zoom_size+1)
                y1_CH3, y2_CH3 = max(0, y_CH3 - zoom_size), min(image_3d.shape[0], y_CH3 + zoom_size+1)

                zoomed_image_CH1 = image_orig[y1_CH1:y2_CH1, x1_CH1:x2_CH1]
                zoomed_image_CH2 = image_orig[y1_CH2:y2_CH2, x1_CH2:x2_CH2]
                zoomed_image_CH3 = image_orig[y1_CH3:y2_CH3, x1_CH3:x2_CH3]

                ax_zoom_CH1.clear()
                ax_zoom_CH1.imshow(zoomed_image_CH1, cmap="gray")
                ax_zoom_CH1.set_xticks([])
                ax_zoom_CH1.set_yticks([])
                ax_zoom_CH1.set_title(f"Zoomed In ({y_CH1}, {x_CH2})")
                rect1 = patches.Rectangle((x1_CH1, y1_CH1), x2_CH1 - x1_CH1, y2_CH1 - y1_CH1, linewidth=1.5, edgecolor='g', facecolor='none')
                ax.add_patch(rect1)
                ax_zoom_CH2.clear()
            
                
                ax_zoom_CH2.imshow(zoomed_image_CH2, cmap="gray")
                ax_zoom_CH2.set_xticks([])
                ax_zoom_CH2.set_yticks([])
                ax_zoom_CH2.set_title(f"Zoomed In ({y_CH2}, {x_CH2})")
                rect2 = patches.Rectangle((x1_CH2, y1_CH2), x2_CH2 - x1_CH2, y2_CH2 - y1_CH2, linewidth=1.5, edgecolor='b', facecolor='none')
                ax.add_patch(rect2)

                ax_zoom_CH3.clear()
                ax_zoom_CH3.imshow(zoomed_image_CH3, cmap="gray")
                ax_zoom_CH3.set_xticks([])
                ax_zoom_CH3.set_yticks([])
                ax_zoom_CH3.set_title(f"Zoomed In ({y_CH3}, {x_CH3})")
                rect3 = patches.Rectangle((x1_CH3, y1_CH3), x2_CH3 - x1_CH3, y2_CH3 - y1_CH3, linewidth=1.5, edgecolor='purple', facecolor='none')
                ax.add_patch(rect3)

    annot.set_visible(visible)
    fig.canvas.draw_idle()


fig, ax = plt.subplots(figsize=(9, 9))
ax.set_position([0.3, 0.3, 0.4, 0.6]) #[left, bottom, width, height]
ax.imshow(image_copy)
ax.grid()

scat1 = ax.scatter(out_pair_arr_CH1[:,1], out_pair_arr_CH1[:,0], s=50, facecolors='none', edgecolors='g', alpha=0)
scat2 = ax.scatter(out_pair_arr_CH2[:,1], out_pair_arr_CH2[:,0], s=50, facecolors='none', edgecolors='b', alpha=0)
scat3 = ax.scatter(out_pair_arr_CH3[:,1], out_pair_arr_CH3[:,0], s=50, facecolors='none', edgecolors='r', alpha=0)
ax.set_title("Mapped Peaks: Click to Zoom in on a Peak")

scatter_data = [(scat1, out_pair_arr_CH1, "CH1"), (scat2, out_pair_arr_CH2, "CH2"), (scat3, out_pair_arr_CH3, "CH3")]

annot = init_annot(ax=ax)

fig.canvas.mpl_connect("button_press_event", lambda event: on_hover_display_three(event, fig, ax, scatter_data, image_copy, hel1_avg_image, zoom_size=6))
fig.canvas.mpl_connect("motion_notify_event", lambda event: on_hover_display_three(event, fig, ax, scatter_data, image_copy, hel1_avg_image, zoom_size=6))

plt.show()