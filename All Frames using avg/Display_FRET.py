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

# This code is substituted for plot_circle(image, 4, y_centre, x_centre, image.shape[0])
circle_array_CH1 = draw_circle(4, poly_pair_arr_CH1_tol4_10[:,1], poly_pair_arr_CH1_tol4_10[:,0], image.shape[0])
circle_array_CH2 = draw_circle(4, poly_pair_arr_CH2_tol4_10[:,1], poly_pair_arr_CH2_tol4_10[:,0], image.shape[0])
circle_array_new = circle_array_CH1 + circle_array_CH2
mask_new = (circle_array_new == [255, 255, 0]).all(axis=-1)
if image.ndim == 2:
    image_3d = np.repeat(image[..., np.newaxis], 3, -1)
elif image.ndim==3 and image.shape[2]==3:
    image_3d = image
image_3d[mask_new] = [255, 255, 0]

def on_hover_intensity(event, pma_file_path, fig, ax, scatter_data, image_3d, tpf=1/100, Intense_axes_CH1=[0.48, 0.7, 0.5, 0.15], Intense_axes_CH2=[0.48, 0.45, 0.5, 0.15], FRET_axes=[0.48, 0.20, 0.5, 0.15]):
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

                ax_intensity_CH1= fig.add_axes(Intense_axes_CH1)
                ax_intensity_CH2= fig.add_axes(Intense_axes_CH2)
                ax_FRET = fig.add_axes(FRET_axes)
                #change y and x of cooresponding peak in other channel as the same index peak in other channel

                if label == "CH1":
                    y, x = scatter_data[0][1][idx]
                    x1, x2 = max(0, x - zoom_size), min(image_3d.shape[1], x + zoom_size)
                    y1, y2 = max(0, y - zoom_size), min(image_3d.shape[0], y + zoom_size)
                    y_CH2, x_CH2 = scatter_data[1][1][idx]
                    x1_CH2, x2_CH2 = max(0, x_CH2 - zoom_size), min(image_3d.shape[1], x_CH2 + zoom_size)
                    y1_CH2, y2_CH2 = max(0, y_CH2 - zoom_size), min(image_3d.shape[0], y_CH2 + zoom_size)


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
                    ax_intensity_CH1.clear()
                    ax_intensity_CH1.plot(time, tot_intensity_all_frames_CH1, color='b', label='CH1')
                    ax_intensity_CH1.set_title(f"Intensity v Time in Donor Peak {idx}")
                    # ax_intensity_CH1.set_xlabel('Time (s)')
                    ax_intensity_CH1.set_ylabel('Intensity')

                    ax_intensity_CH2.clear()               
                    ax_intensity_CH2.plot(time, tot_intensity_all_frames_CH2, color='g', label='CH2')
                    ax_intensity_CH2.set_title(f"Intensity v Time in Acceptor Peak {idx}")
                    # ax_intensity_CH2.set_xlabel('Time (s)')
                    ax_intensity_CH2.set_ylabel('Intensity')

                    FRET_values = calc_FRET(tot_intensity_all_frames_CH1, tot_intensity_all_frames_CH2)
                    ax_FRET.clear()               
                    ax_FRET.plot(time, FRET_values, color='r')
                    ax_FRET.set_title(f"FRET v Time in Pair {idx}")
                    ax_FRET.set_xlabel('Time (s)')
                    ax_FRET.set_ylabel('FRET Efficiency')

                    rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='b', facecolor='none')
                    ax.add_patch(rect1)
                    rect2 = patches.Rectangle((x1_CH2, y1_CH2), x2_CH2 - x1_CH2, y2_CH2 - y1_CH2, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect2)
                else:
                    ax_intensity_CH1.clear()
                    ax_intensity_CH2.clear()
                    y, x = scatter_data[1][1][idx]
                    x1, x2 = max(0, x - zoom_size), min(image_3d.shape[1], x + zoom_size)
                    y1, y2 = max(0, y - zoom_size), min(image_3d.shape[0], y + zoom_size)
                    y_CH1, x_CH1 = scatter_data[0][1][idx]
                    x1_CH1, x2_CH1 = max(0, x_CH1- zoom_size), min(image_3d.shape[1], x_CH1 + zoom_size)
                    y1_CH1, y2_CH1 = max(0, y_CH1 - zoom_size), min(image_3d.shape[0], y_CH1 + zoom_size)

                    
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
                    ax_intensity_CH2.plot(time, tot_intensity_all_frames_CH2, color='g', label='CH2')
                    ax_intensity_CH2.set_title(f"Intensity v Time in Acceptor Peak {idx}")
                    # ax_intensity_CH2.set_xlabel('Time (s)')
                    ax_intensity_CH2.set_ylabel('Intensity')

                    ax_intensity_CH1.plot(time, tot_intensity_all_frames_CH1, color='b', label='CH1')
                    ax_intensity_CH1.set_title(f"Intensity v Time in Donor Peak {idx}")
                    # ax_intensity_CH1.set_xlabel('Time (s)')
                    ax_intensity_CH1.set_ylabel('Intensity')

                    FRET_values = calc_FRET(tot_intensity_all_frames_CH1, tot_intensity_all_frames_CH2)
                    ax_FRET.clear()               
                    ax_FRET.plot(time, FRET_values, color='r')
                    ax_FRET.set_title(f"FRET v Time in Pair {idx}")
                    ax_FRET.set_xlabel('Time (s)')
                    ax_FRET.set_ylabel('FRET Efficiency')

                    rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
                    ax.add_patch(rect2)
                    rect1 = patches.Rectangle((x1_CH1, y1_CH1), x2_CH1 - x1_CH1, y2_CH1 - y1_CH1, linewidth=2, edgecolor='b', facecolor='none')
                    ax.add_patch(rect1)

    annot.set_visible(visible)
    fig.canvas.draw_idle()


# Create main figure

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
#
ax.set_position([0.03, 0.2, 0.4, 0.6])
ax.imshow(image_3d)

scat1 = ax.scatter(poly_pair_arr_CH1_tol4_10[:,1], poly_pair_arr_CH1_tol4_10[:,0], s=50, facecolors='none', edgecolors='b', alpha=0)
scat2 = ax.scatter(poly_pair_arr_CH2_tol4_10[:,1], poly_pair_arr_CH2_tol4_10[:,0], s=50, facecolors='none', edgecolors='g', alpha=0)
ax.set_title("Mapped Peaks: Click For Intensity v Time")

scatter_data = [(scat1, poly_pair_arr_CH1_tol4_10 , "CH1"), (scat2, poly_pair_arr_CH2_tol4_10 , "CH2")]

annot = init_annot(ax=ax)

fig.canvas.mpl_connect("button_press_event", lambda event: on_hover_intensity(event, file_path, fig, ax, scatter_data, image_3d))
fig.canvas.mpl_connect("motion_notify_event", lambda event: on_hover_intensity(event, file_path, fig, ax, scatter_data, image_3d))

plt.show()