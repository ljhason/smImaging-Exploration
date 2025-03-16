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

# fig = plt.figure(figsize=(8,8))
# ax = fig.subplots()
# plt.imshow(image_3d)
# scat1 = ax.scatter(poly_pair_arr_CH1_tol4_10[:,1], poly_pair_arr_CH1_tol4_10[:,0], s=50, facecolors='none', edgecolors='b', alpha=0)
# scat2 = ax.scatter(poly_pair_arr_CH2_tol4_10[:,1], poly_pair_arr_CH2_tol4_10[:,0], s=50, facecolors='none', edgecolors='r', alpha=0)
# plt.title('PolyMap circle display')

# def display_peak_trigger(event, scatter_data):
#     """ Checks if the mouse hovers over a point and updates annotation """
#     visible = False
#     for scatter, peaks, label in scatter_data:
#         cont, ind = scatter.contains(event)  # Check if cursor is over a scatter point
#         if cont:
#             update_annot(ind, scatter, peaks, label)  # Update annotation
#             visible = True
#             if event.name == "button_press_event":
#                 # Display a zoomed in version of the peak figure (10 pixels in all directions)
#                 zoomed_in_peak = peaks[ind["ind"][0]]
#                 plot_circle(image, 10, zoomed_in_peak[1], zoomed_in_peak[0], image.shape[0])
#                 plt.show()
#             break

#     annot.set_visible(visible)  # Show annotation only if hovering over a point
#     fig.canvas.draw_idle()  # Redraw figure to update annotation

# annot = init_annot(ax=ax)

# #must define scat1 and scat 2!!
# scatter_data = [(scat1, poly_pair_arr_CH1_tol4_10 , "CH1"), (scat2, poly_pair_arr_CH2_tol4_10 , "CH2")]
# fig.canvas.mpl_connect("motion_notify_event", lambda event: display_peak_trigger(event, scatter_data))
# fig.canvas.mpl_connect("button_press_event", lambda event: display_peak_trigger(event, scatter_data))

# plt.show()

# Create main figure
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image_3d)
ax.set_title("Click to Zoom In")

# Create an inset zoomed-in axis
zoom_size = 5  # Size of zoomed-in region
# place ax_zoom in the top right corner of the figure

ax_zoom_CH1 = fig.add_axes([0.75, 0.6, 0.2, 0.2])
ax_zoom_CH1.set_xticks([])
ax_zoom_CH1.set_yticks([])
ax_zoom_CH1.set_title("Zoomed In CH1")

# ax_zoom_CH2 = fig.add_axes([0.75, 0.3, 0.2, 0.2])
# ax_zoom_CH2.set_xticks([])
# ax_zoom_CH2.set_yticks([])
# ax_zoom_CH2.set_title("Zoomed In CH2")

def zoom_trigger(event):
    """ Handles mouse click event to zoom in on the clicked point. """
    if event.inaxes != ax:  # Ensure click is within the main image
        return
    
    x_CH1, y_CH1 = int(event.xdata), int(event.ydata)  # Get click coordinates
    
    # Define zoom region (clipping edges if necessary)
    x1_CH1, x2_CH1 = max(0, x_CH1 - zoom_size), min(image_3d.shape[1], x_CH1 + zoom_size)
    y1_CH1, y2_CH1 = max(0, y_CH1 - zoom_size), min(image_3d.shape[0], y_CH1 + zoom_size)

    # Extract and display the zoomed-in portion
    zoomed_image = image_3d[y1_CH1:y2_CH1, x1_CH1:x2_CH1]
    ax_zoom_CH1.clear()
    ax_zoom_CH1.imshow(zoomed_image, cmap="gray")
    ax_zoom_CH1.set_xticks([])
    ax_zoom_CH1.set_yticks([])
    ax_zoom_CH1.set_title(f"Zoomed In ({y1_CH1}:{y2_CH1}, {x1_CH1}:{x2_CH1})")
    # could try adding a feature where if the up, down, left or right buttons are clicked the zoom region changes
    fig.canvas.draw_idle()  # Redraw figure

# Connect click event to function
fig.canvas.mpl_connect("button_press_event", zoom_trigger)

plt.show()