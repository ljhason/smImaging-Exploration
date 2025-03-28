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

# Poly Mapping
CH1_peaks_10 = np.array([[18,92], [16,213], [108,43], [106, 176], [210,51], [234, 219], [366,12], [322,192], [478, 106], [502,160]])
CH2_peaks_10_new = np.array([[22,349],[19,470],[111,300],[108,433],[212,307],[234,475],[367,268],[321,448],[476,361],[499,414]])
params_x_man_10_new, params_y_man_10_new = find_polyfit_params(CH1_peaks_10, CH2_peaks_10_new, degree=3)
mapped_peaks_10_new = apply_polyfit_params(good_peaks_1, params_x_man_10_new, params_y_man_10_new).astype(np.uint16)
poly_pair_count_tol4_10_new, poly_pair_arr_CH1_tol4_10_new, poly_pair_arr_CH2_tol4_10_new = find_polyfit_pairs(mapped_peaks_10_new, good_peaks_1, tolerance=3)

poly_pair_arr_CH2_tol4_10_new_unshift = shift_peaks(poly_pair_arr_CH2_tol4_10_new, [1, 10])
poly_pair_arr_CH2_tol4_10_curr = poly_pair_arr_CH2_tol4_10_new_unshift[poly_pair_arr_CH2_tol4_10_new_unshift[:,1] <= 502]
poly_pair_arr_CH1_tol4_10_curr = poly_pair_arr_CH1_tol4_10_new[poly_pair_arr_CH2_tol4_10_new_unshift[:,1] <= 502]
# This code is substituted for plot_circle(image, 4, y_centre, x_centre, image.shape[0])
circle_array_CH1 = draw_circle(4, poly_pair_arr_CH1_tol4_10_curr[:,1], poly_pair_arr_CH1_tol4_10_curr[:,0], image.shape[0])
circle_array_CH2 = draw_circle(4, poly_pair_arr_CH2_tol4_10_curr[:,1], poly_pair_arr_CH2_tol4_10_curr[:,0], image.shape[0])
circle_array_new = circle_array_CH1 + circle_array_CH2
mask_new = (circle_array_new == [255, 255, 0]).all(axis=-1)
if image.ndim == 2:
    image_3d = np.repeat(image[..., np.newaxis], 3, -1)
elif image.ndim==3 and image.shape[2]==3:
    image_3d = image
image_3d[mask_new] = [255, 255, 0]

def on_hover_intensity(event, pma_file_path, fig, ax, scatter_data, image_3d, tpf=1/100, R_0=56, Intense_axes=[0.48, 0.6, 0.5, 0.3], FRET_axes=[0.48, 0.35, 0.5, 0.15], dist_axes=[0.48, 0.1, 0.5, 0.15], CH1_zoom_axes=[0.04, 0.06, 0.15, 0.15], CH2_zoom_axes=[0.27, 0.06, 0.15, 0.15]):
    """ Checks if the mouse hovers over a point and updates annotation """
    visible = False
    zoom_size=6
    for scatter, peaks, label in scatter_data:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, scatter, peaks, label)
            visible = True
            idx = ind['ind'][0]

            if event.name == "button_press_event":
                Frames_data = read_pma(pma_file_path)
                for patch in ax.patches:
                    patch.remove()
                    
                for ax_zoom in fig.axes:
                    if ax_zoom is not ax:  # Keep the main axis
                        fig.delaxes(ax_zoom)
                ax_zoom_CH1 = fig.add_axes(CH1_zoom_axes)
                ax_zoom_CH2 = fig.add_axes(CH2_zoom_axes)
                ax_intensity= fig.add_axes(Intense_axes)
                ax_FRET = fig.add_axes(FRET_axes)
                ax_dist = fig.add_axes(dist_axes)
                #change y and x of cooresponding peak in other channel as the same index peak in other channel

                if label == "CH1":
                    y, x = scatter_data[0][1][idx]
                    x1, x2 = max(0, x - zoom_size), min(image_3d.shape[1], x + zoom_size)
                    y1, y2 = max(0, y - zoom_size), min(image_3d.shape[0], y + zoom_size)
                    y_CH2, x_CH2 = scatter_data[1][1][idx]
                    x1_CH2, x2_CH2 = max(0, x_CH2 - zoom_size), min(image_3d.shape[1], x_CH2 + zoom_size)
                    y1_CH2, y2_CH2 = max(0, y_CH2 - zoom_size), min(image_3d.shape[0], y_CH2 + zoom_size)
                    zoomed_image = image_3d[y1:y2, x1:x2]
                    ax_zoom_CH1.clear()
                    ax_zoom_CH1.imshow(zoomed_image, cmap="gray")
                    ax_zoom_CH1.set_xticks([])
                    ax_zoom_CH1.set_yticks([])
                    ax_zoom_CH1.set_title("")
                    ax_zoom_CH1.set_title(f"Zoomed In ({y}, {x})")
                    zoomed_image_CH2 = image_3d[y1_CH2:y2_CH2, x1_CH2:x2_CH2]
                    ax_zoom_CH2.imshow(zoomed_image_CH2, cmap="gray")
                    ax_zoom_CH2.set_xticks([])
                    ax_zoom_CH2.set_yticks([])
                    ax_zoom_CH2.set_title(f"Zoomed In ({y_CH2}, {x_CH2})")
                                        

                    tot_intensity_all_frames_CH1 = []
                    tot_intensity_all_frames_CH2 = []

                    for i in range(len(Frames_data)): #for i in range(795): i= 0, 1, 2,..., 794

                        # transforms from 2D to 3D
                        if Frames_data[i].ndim == 2:
                            frame_3d = np.repeat(Frames_data[i][..., np.newaxis], 3, -1)
                        elif Frames_data[i].ndim==3 and Frames_data[i].shape[2]==3:
                            frame_3d = Frames_data[i]
                        frame_3d[mask_new] = [255, 255, 0]

                        total_intensity_CH1,_ = intensity_in_circle(frame_3d, 4, y, x)
                        total_intensity_CH2,_ = intensity_in_circle(frame_3d, 4, y_CH2, x_CH2)
                        tot_intensity_all_frames_CH1.append(total_intensity_CH1)
                        tot_intensity_all_frames_CH2.append(total_intensity_CH2)
                    
                    time = np.arange(0, len(tot_intensity_all_frames_CH1)/100, tpf)
                    ax_intensity.clear()
                    ax_intensity.plot(time, tot_intensity_all_frames_CH1, color='b', label='CH1')
                    ax_intensity.plot(time, tot_intensity_all_frames_CH2, color='g', label='CH2')
                    ax_intensity.set_title(f"Intensity v Time in  Peak {idx}")
                    ax_intensity.set_xlabel('Time (s)')
                    ax_intensity.set_ylabel('Intensity')
                    ax_intensity.set_ylim(-255, max(max(tot_intensity_all_frames_CH1), max(tot_intensity_all_frames_CH2))+500)
                    ax_intensity.legend(bbox_to_anchor=(1.0, 1.2), loc='upper right')

                    FRET_values = calc_FRET(tot_intensity_all_frames_CH1, tot_intensity_all_frames_CH2)
                    ax_FRET.clear()               
                    ax_FRET.plot(time, FRET_values, color='r')
                    ax_FRET.set_title(f"FRET v Time in Pair {idx}")
                    ax_FRET.set_xlabel('Time (s)')
                    ax_FRET.set_ylabel('FRET Efficiency')

                    dist_values = calc_distance(FRET_values, R_0)
                    ax_dist.clear()
                    ax_dist.plot(time, dist_values, color='y')
                    ax_dist.set_title(f"Distance v Time in Pair {idx}")
                    ax_dist.set_xlabel('Time (s)')
                    ax_dist.set_ylabel('Distance')

                    rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='b', facecolor='none')
                    ax.add_patch(rect1)
                    rect2 = patches.Rectangle((x1_CH2, y1_CH2), x2_CH2 - x1_CH2, y2_CH2 - y1_CH2, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect2)
                else:

                    ax_intensity.clear()
                    y, x = scatter_data[1][1][idx]
                    x1, x2 = max(0, x - zoom_size), min(image_3d.shape[1], x + zoom_size)
                    y1, y2 = max(0, y - zoom_size), min(image_3d.shape[0], y + zoom_size)
                    y_CH1, x_CH1 = scatter_data[0][1][idx]
                    x1_CH1, x2_CH1 = max(0, x_CH1- zoom_size), min(image_3d.shape[1], x_CH1 + zoom_size)
                    y1_CH1, y2_CH1 = max(0, y_CH1 - zoom_size), min(image_3d.shape[0], y_CH1 + zoom_size)
                    zoomed_image = image_3d[y1:y2, x1:x2]
                    ax_zoom_CH2.imshow(zoomed_image, cmap="gray")
                    ax_zoom_CH2.set_xticks([])
                    ax_zoom_CH2.set_yticks([])
                    ax_zoom_CH2.set_title(f"Zoomed In ({y}, {x})")
                    zoomed_image_CH1 = image_3d[y1_CH1:y2_CH1, x1:x2_CH1]
                    ax_zoom_CH1.imshow(zoomed_image_CH1, cmap="gray")
                    ax_zoom_CH1.set_xticks([])
                    ax_zoom_CH1.set_yticks([])
                    ax_zoom_CH1.set_title(f"Zoomed In ({y_CH1}, {x_CH1})")
                    tot_intensity_all_frames_CH1 = []
                    tot_intensity_all_frames_CH2 = []


                    for i in range(len(Frames_data)): #for i in range(795): i= 0, 1, 2,..., 794
                        
                        # transforms from 2D to 3D
                        if Frames_data[i].ndim == 2:
                            frame_3d = np.repeat(Frames_data[i][..., np.newaxis], 3, -1)
                        elif Frames_data[i].ndim==3 and Frames_data[i].shape[2]==3:
                            frame_3d = Frames_data[i]
                        
                        frame_3d[mask_new] = [255, 255, 0]
                        total_intensity_CH2,_ = intensity_in_circle(frame_3d, 4, y, x)
                        total_intensity_CH1,_ = intensity_in_circle(frame_3d, 4, y_CH1, x_CH1)
                        tot_intensity_all_frames_CH2.append(total_intensity_CH2)
                        tot_intensity_all_frames_CH1.append(total_intensity_CH1)

                    time = np.arange(0, len(tot_intensity_all_frames_CH2)/100, tpf)
                    ax_intensity.plot(time, tot_intensity_all_frames_CH2, color='g', label='CH2')
                    ax_intensity.plot(time, tot_intensity_all_frames_CH1, color='b', label='CH1')
                    ax_intensity.set_title(f"Intensity v Time in Acceptor Peak {idx}")
                    ax_intensity.set_xlabel('Time (s)')
                    ax_intensity.set_ylabel('Intensity')
                    ax_intensity.legend(bbox_to_anchor=(1.0, 1.2), loc='upper right')

                    FRET_values = calc_FRET(tot_intensity_all_frames_CH1, tot_intensity_all_frames_CH2)
                    ax_FRET.clear()               
                    ax_FRET.plot(time, FRET_values, color='r')
                    ax_FRET.set_title(f"FRET v Time in Pair {idx}")
                    ax_FRET.set_xlabel('Time (s)')
                    ax_FRET.set_ylabel('FRET Efficiency')

                    dist_values = calc_distance(FRET_values, R_0)
                    ax_dist.clear()
                    ax_dist.plot(time, dist_values, color='y')
                    ax_dist.set_title(f"Distance v Time in Pair {idx}")
                    ax_dist.set_xlabel('Time (s)')
                    ax_dist.set_ylabel('Distance')

                    rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect2)
                    rect1 = patches.Rectangle((x1_CH1, y1_CH1), x2_CH1 - x1_CH1, y2_CH1 - y1_CH1, linewidth=2, edgecolor='b', facecolor='none')
                    ax.add_patch(rect1)

    annot.set_visible(visible)
    fig.canvas.draw_idle()


# Create main figure

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.set_position([0.01, 0.3, 0.4, 0.6]) #[left, bottom, width, height]
ax.imshow(image_3d)
ax.grid()

scat1 = ax.scatter(poly_pair_arr_CH1_tol4_10_curr[:,1], poly_pair_arr_CH1_tol4_10_curr[:,0], s=50, facecolors='none', edgecolors='b', alpha=0)
scat2 = ax.scatter(poly_pair_arr_CH2_tol4_10_curr[:,1], poly_pair_arr_CH2_tol4_10_curr[:,0], s=50, facecolors='none', edgecolors='g', alpha=0)
ax.set_title("Mapped Peaks: Click For Plots")

scatter_data = [(scat1, poly_pair_arr_CH1_tol4_10_curr , "CH1"), (scat2, poly_pair_arr_CH2_tol4_10_curr , "CH2")]

annot = init_annot(ax=ax)

fig.canvas.mpl_connect("button_press_event", lambda event: on_hover_intensity(event, file_path, fig, ax, scatter_data, image_3d))
fig.canvas.mpl_connect("motion_notify_event", lambda event: on_hover_intensity(event, file_path, fig, ax, scatter_data, image_3d))

plt.show()
