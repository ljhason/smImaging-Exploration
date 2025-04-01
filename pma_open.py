import sys
import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy.ndimage import maximum_filter, label
from skimage import io, feature, draw
from skimage import color
from PIL import Image
from skimage.util.shape import view_as_blocks
from skimage.feature import peak_local_max
from matplotlib.widgets import Cursor, Button
from matplotlib import patches

def read_pma_f0(pma_file_path):
    try:
        with open(pma_file_path, "rb") as f:
            #Assign X_pixels and Y_pixels as the first two 16-bit integers in the file
            #<:little-endian (least significant byte first), HH:two 16-bit integers
            X_pixels, Y_pixels = struct.unpack("<HH", f.read(4))
            print(f"Image Size: {X_pixels} x {Y_pixels}")
            
            #Calc number of frames
            f.seek(0, 2) #sets pointer to end of file .seek(offset, from_what)
            filesize = f.tell() #returns current (end) position of pointer
            # Nframes = (filesize - 4) // (X_pixels * Y_pixels)  #Assuming 4-byte header
            f.seek(0, 4) #Reset file pointer to immediately after 4 byte header

            #Read the binary image data
            frame_data0 = f.read(X_pixels * Y_pixels)

            image_data = np.frombuffer(frame_data0, dtype=np.uint8).reshape((Y_pixels, X_pixels))

            return image_data

    except Exception as e:
        print(f"Error reading .pma file: {e}")
        return None

def read_pma(pma_file_path):
    try:
        with open(pma_file_path, "rb") as f:
            # Assign X_pixels and Y_pixels as the first two 16-bit integers in the file
            X_pixels, Y_pixels = struct.unpack("<HH", f.read(4))
            print(f"Image Size: {X_pixels} x {Y_pixels}")
            
            # Calculate number of frames
            f.seek(0, 2)  # sets pointer to end of file
            filesize = f.tell()  # returns current (end) position of pointer
            Nframes = (filesize - 4) // (X_pixels * Y_pixels)  # Assuming 4-byte header
            f.seek(4)  # Reset file pointer to immediately after 4 byte header
            return [np.frombuffer(f.read(X_pixels*Y_pixels), dtype=np.uint8).reshape((Y_pixels, X_pixels)) for frame_idx in range(Nframes)]

    except Exception as e:
        print(f"Error reading .pma file: {e}")
        return None
    
def generate_images(pma_file_path):
    try:
        output_name = pma_file_path.split(".")[-2].split("/")[-1]
        if not os.path.exists(f"{output_name}_Files"):
            os.makedirs(f"{output_name}_Files")
        else:
            print(f"Directory already exists: {output_name}_Files")
            return None
        
        Frames_data = read_pma(pma_file_path)
        for frame_idx, frame_data in enumerate(Frames_data):
            plt.imsave(f"{output_name}_Files/{output_name}frame_{frame_idx}.png", frame_data, cmap='gray')

    except Exception as e:
        print(f"Error generating images or creating directory: {e}")
        return None
    

def generate_mp4(images_path, fps=100):
    try:
        pma_name = f"{images_path.split('_')[-2]}"
        video_name = f"{pma_name}.mp4"
        video_file= os.path.join(images_path, f"{pma_name}_Video")
        
        if not os.path.exists(video_file):
            os.makedirs(video_file)
        else:
            print(f"Directory already exists: {video_file}")
            return None
            
        images = [img for img in os.listdir(images_path) if img.endswith(".png")]
        images.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        frame = cv2.imread(os.path.join(images_path, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(os.path.join(video_file, video_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(images_path, image)))

        video.release()
        cv2.destroyAllWindows()

        print(f"Video sucessfully generated and saved as: {video_name}")
        print(f"Images: {(images)}")
    
    except Exception as e:
        print(f"Error generating video: {e}")
        return None
    
def avg_frame_arr(pma_file_path):
    try:

        Frames_data = read_pma(pma_file_path)
        avg_frame_data = np.mean(Frames_data, axis=0).astype(np.uint8)
        print(f"Sucessfully generated average frame")
        return avg_frame_data

    except Exception as e:
        print(f"Error generating average frame: {e}")
        return None


def avg_frame_png(pma_file_path):
    try:
        avg_frame_data = avg_frame_arr(pma_file_path)
        output_name = pma_file_path.split(".")[-2].split("/")[-1]
        image_file_name = f'{output_name}_Avg_Frame.png'
        if not os.path.exists(f"{output_name}_Avg_Frame"):
            os.makedirs(f"{output_name}_Avg_Frame")
        else:
            pass
        image = Image.fromarray(avg_frame_data)
        image.save(f"{output_name}_Avg_Frame/{image_file_name}")
        print(f"Average frame saved as: {image_file_name}")

    except Exception as e:
        print(f"Error generating average frame: {e}")
        return None

def dim_to_3(image):
    return np.stack((image,) * 3, axis=-1)

def find_peaks_scipy_IDL(image_path, sigma=3, block_size=16, scaler_percent=32):
    std = 4*sigma
    # Load image (assumes grayscale uint8 image)
    image = io.imread(image_path, as_gray=True).astype(np.uint8)
    height, width = image.shape
    image_1 = image.copy()
    min_intensity = np.min(image_1)
    max_intensity = np.max(image_1)
    threshold = min_intensity + (scaler_percent / 100.0) * (max_intensity - min_intensity)
        
    background = np.zeros((height, width), dtype=np.float32)

    for i in range(8, height, block_size):
        for j in range(8, width, block_size):
            background[(i-8)//block_size, (j-8)//block_size] = np.min(image_1[i-8:i+8, j-8:j+8])
        
    # Subtract background
    background = np.clip(background.astype(np.uint8) - 10, 0, 255)
    image_1 = image - background
        
    image_2 = image_1.copy()
    med = np.median(image_1)

    # Apply threshold
    image_2[image_2 < (med + 3*std)] = 0
    
    # Detect peaks using peak_local_max
    peak_coords = peak_local_max(image_2, min_distance=int(sigma), threshold_abs=threshold)
    
    return peak_coords, image_2

#Same Good Peaks as in img_avg.ipynb file
def good_peak_finder(image_path, sigma=3, block_size=16, scaler_percent=32, boarder=10, max_rad=3):
    peaks_coords_IDL, image_2 = find_peaks_scipy_IDL(image_path, sigma, block_size, scaler_percent)
    large_peaks = []
    correct_size_peaks = []
    height, width = io.imread(image_path).shape

    for peak in peaks_coords_IDL:
        y, x = peak
        # Extract the peak region, if pixels outside of 5x5 region are non-zero, then append peak to large_peaks
        if image_2[y, x + max_rad+1] > 0 or image_2[y, x - max_rad] > 0 or image_2[y+max_rad+1, x ] > 0 or image_2[y-max_rad, x] > 0 or peak[0] < boarder or peak[0] > height - boarder or peak[1] < boarder or peak[1] > width - boarder:
            large_peaks.append(peak)
        else:
            correct_size_peaks.append(peak)

    correct_size_peaks = np.array(correct_size_peaks)
    large_peaks = np.array(large_peaks)
    
    return correct_size_peaks, large_peaks

def shift_peaks(peaks, shift=[0, 256]):
    return np.add(peaks, shift)

#change the arrow colour to white
def init_annot(ax, text="", xy=(0, 0), xytext=(0, 10),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->", color="w")):
    global annot
    annot = ax.annotate(text, xy=xy, xytext=xytext, textcoords=textcoords, bbox=bbox, arrowprops=arrowprops)
    annot.set_visible(False)
    return annot

# Function to update annotation text and position
def update_annot(ind, scatter, peaks, label):
    """ Updates the annotation position and text """
    idx = ind["ind"][0]
    y, x = peaks[idx]
    annot.xy = (scatter.get_offsets()[idx][0], scatter.get_offsets()[idx][1])
    annot.set_text(f"{label} Peak {idx}: (y, x) = ({y}, {x})")
    annot.set_visible(True)

def print_coords_trigger(event, fig, scatter_data):
    """ Checks if the mouse hovers over a point and updates annotation """
    visible = False
    for scatter, peaks, label in scatter_data:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind, scatter, peaks, label)
            visible = True
            if event.name == "button_press_event":
        
                print(f"{label}_Peak{ind['ind'][0]} (y, x):({peaks[ind['ind'][0]][0]},{peaks[ind['ind'][0]][1]})")
            break

    annot.set_visible(visible)
    fig.canvas.draw_idle()

def find_linear_pairs(peaks_1, peaks_2, tolerance=1, width = 512):
    # peaks_2 coordinates goes from [0, 512] to [256,512]
    gp1_list = [tuple(peak) for peak in peaks_1]
    gp2_list = [tuple(peak) for peak in peaks_2]
    gp2_set = set(gp2_list)
    linear_pair_count = 0
    linear_pair_arr_CH1 = []
    linear_pair_arr_CH2 = []
    try: 
        if width == 512:
            for coord in gp1_list:
                    for c in gp2_set:
                        if (abs(coord[0] - c[0])) <=tolerance and (256-tolerance <= abs(coord[1] - c[1]) <= 256+tolerance) and c not in linear_pair_arr_CH2:
                            linear_pair_count += 1
                            linear_pair_arr_CH1.append(coord)
                            linear_pair_arr_CH2.append(c)
                            break
        elif width == 256:
            for coord in gp1_list:
                    for c in gp2_set:
                        if (abs(coord[0] - c[0])) <=tolerance and (abs(coord[1] - c[1]) <= tolerance) and c not in linear_pair_arr_CH2:
                            linear_pair_count += 1
                            linear_pair_arr_CH1.append(coord)
                            linear_pair_arr_CH2.append(c)
                            break
    except Exception as e:
        print(f"Error finding linear pairs: {e}")
        return None
    linear_pair_arr_CH1 = np.array(linear_pair_arr_CH1)
    linear_pair_arr_CH2 = np.array(linear_pair_arr_CH2)
    return linear_pair_count, linear_pair_arr_CH1, linear_pair_arr_CH2


def find_polyfit_params(peaks_1, peaks_2, degree=2):
    y1, x1 = peaks_1[:, 0], peaks_1[:, 1] 
    y2, x2 = peaks_2[:, 0], peaks_2[:, 1] 

    # Fit polynomials for x and y separately
    params_x = np.polyfit(x1, x2, degree)  # Fit x transformation
    params_y = np.polyfit(y1, y2, degree)  # Fit y transformation

    return params_x, params_y  # Returns polynomial coefficients

def apply_polyfit_params(CH1_peaks, params_x, params_y):
    y1, x1 = CH1_peaks[:, 0], CH1_peaks[:, 1]
    x_mapped = np.polyval(params_x, x1)  # Apply X transformation
    y_mapped = np.polyval(params_y, y1)  # Apply Y transformation
    return np.column_stack((y_mapped, x_mapped))  # Return transformed points

def find_polyfit_pairs(mapped_peaks, peaks_1, tolerance=1):
    # we are comparing mapped peaks to CH2 peaks
    map_list = [tuple(peak) for peak in mapped_peaks]
    gp1_list = [tuple(peak) for peak in peaks_1]
    gp1_set = set(gp1_list)

    poly_pair_count = 0
    poly_pair_arr_CH1 = []
    poly_pair_arr_CH2 = []

    for coord in map_list:
            for c in gp1_set:
                if (abs(coord[0] - c[0])) <=tolerance and (256-tolerance <= abs(coord[1] - c[1]) <= 256+tolerance) and c not in poly_pair_arr_CH1:
                    poly_pair_count += 1
                    poly_pair_arr_CH1.append(c)
                    poly_pair_arr_CH2.append(coord)
                    break
                
    poly_pair_arr_CH1 = np.array(poly_pair_arr_CH1)
    poly_pair_arr_CH2 = np.array(poly_pair_arr_CH2)
    return poly_pair_count, poly_pair_arr_CH1, poly_pair_arr_CH2


# Midpoint circle algorithm 
def draw_circle(radius, y_centre, x_centre, background_dim, colour = [255, 255, 0]):
    circle_array = np.zeros((background_dim, background_dim, 3), dtype=np.uint8)
    # Midpoint circle algorithm
    y = radius
    x = 0
    p = 1 - radius
    
    while y >= x:
        circle_array[y_centre + y, x_centre + x] = colour
        circle_array[y_centre - y, x_centre + x] = colour
        circle_array[y_centre + y, x_centre - x] = colour
        circle_array[y_centre - y, x_centre - x] = colour
        circle_array[y_centre + x, x_centre + y] = colour
        circle_array[y_centre - x, x_centre + y] = colour
        circle_array[y_centre + x, x_centre - y] = colour
        circle_array[y_centre - x, x_centre - y] = colour
         
        x += 1
        if p <= 0:
            p = p + 2 * x + 1
        else:
            y -= 1
            p = p + 2 * x - 2 * y + 1
    
    return circle_array

#changed the arguments, please edit in jupyter scripts!
def plot_circle(image, y_centre, x_centre, colour = [255, 255, 0]):
    circle_array = draw_circle(4, y_centre, x_centre, image.shape[0])
    mask = (circle_array == [255, 255, 0]).all(axis=-1)
    try:
        if image.ndim == 2:
            image_3d = np.repeat(image[..., np.newaxis], 3, -1)
        elif image.ndim==3 and image.shape[2]==3:
            image_3d = image
    except Exception as e:
        print(f"Error plotting circle: {e}")
        return None
    
    # Set the pixels in the mask to be yellow
    image_3d[mask] = colour
    # Display the modified image

    plt.imshow(image_3d)
    plt.show()

#counts the pixels within the circle (should have 45 if the radius is 4)
def count_circle(radius, y_centre=12, x_centre=12):
    total = 0
    #filling in the circle
    for i in range(x_centre - radius, x_centre + radius + 1):
        for j in range(y_centre - radius, y_centre + radius + 1):
            if (i - x_centre) ** 2 + (j - y_centre) ** 2 < radius ** 2:
                total +=1
    
    return total

def on_hover(event, fig, ax, scatter_data, image_3d, image_orig, zoom_size=6,CH1_zoom_axes=[0.75, 0.6, 0.2, 0.2], CH2_zoom_axes=[0.75, 0.3, 0.2, 0.2]):
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

                y_CH1, x_CH1 = scatter_data[0][1][idx]
                x1_CH1, x2_CH1 = max(0, x_CH1 - zoom_size), min(image_3d.shape[1], x_CH1 + zoom_size+1)
                y1_CH1, y2_CH1 = max(0, y_CH1 - zoom_size), min(image_3d.shape[0], y_CH1 + zoom_size+1)
                y_CH2, x_CH2 = scatter_data[1][1][idx]
                x1_CH2, x2_CH2 = max(0, x_CH2 - zoom_size), min(image_3d.shape[1], x_CH2 + zoom_size+1)
                y1_CH2, y2_CH2 = max(0, y_CH2 - zoom_size), min(image_3d.shape[0], y_CH2 + zoom_size+1)

                zoomed_image_CH1 = image_orig[y1_CH1:y2_CH1, x1_CH1:x2_CH1]
                zoomed_image_CH2 = image_orig[y1_CH2:y2_CH2, x1_CH2:x2_CH2]

                ax_zoom_CH1.clear()
                ax_zoom_CH1.imshow(zoomed_image_CH1, cmap="gray")
                ax_zoom_CH1.set_xticks([])
                ax_zoom_CH1.set_yticks([])
                ax_zoom_CH1.set_title("")
                ax_zoom_CH1.set_title(f"Zoomed In ({y_CH1}, {x_CH2})")
                rect1 = patches.Rectangle((x1_CH1, y1_CH1), x2_CH1 - x1_CH1, y2_CH1 - y1_CH1, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect1)
                ax_zoom_CH2.clear()
            
                
                ax_zoom_CH2.imshow(zoomed_image_CH2, cmap="gray")
                ax_zoom_CH2.set_xticks([])
                ax_zoom_CH2.set_yticks([])
                ax_zoom_CH2.set_title(f"Zoomed In ({y_CH2}, {x_CH2})")
                rect2 = patches.Rectangle((x1_CH2, y1_CH2), x2_CH2 - x1_CH2, y2_CH2 - y1_CH2, linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect2)
                

    annot.set_visible(visible)
    fig.canvas.draw_idle()


def on_hover_intensity(event, pma_file_path, fig, ax, scatter_data, image_3d, image_orig, mask, radius=4, tpf=1/50, R_0=56, Intense_axes_CH1=[0.48, 0.81, 0.5, 0.15], Intense_axes_CH2=[0.48, 0.56, 0.5, 0.15], FRET_axes=[0.48, 0.31, 0.5, 0.15], dist_axes=[0.48, 0.06, 0.5, 0.15], CH1_zoom_axes=[0.04, 0.06, 0.15, 0.15], CH2_zoom_axes=[0.22, 0.06, 0.15, 0.15]):
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

                ax_intensity_CH1= fig.add_axes(Intense_axes_CH1)
                ax_intensity_CH2= fig.add_axes(Intense_axes_CH2)
                ax_FRET = fig.add_axes(FRET_axes)
                ax_dist = fig.add_axes(dist_axes)

                y_CH1, x_CH1 = scatter_data[0][1][idx]
                x1_CH1, x2_CH1 = max(0, x_CH1 - zoom_size), min(image_3d.shape[1], x_CH1 + zoom_size)
                y1_CH1, y2_CH1 = max(0, y_CH1 - zoom_size), min(image_3d.shape[0], y_CH1 + zoom_size)
                y_CH2, x_CH2 = scatter_data[1][1][idx]
                x1_CH2, x2_CH2 = max(0, x_CH2 - zoom_size), min(image_3d.shape[1], x_CH2 + zoom_size)
                y1_CH2, y2_CH2 = max(0, y_CH2 - zoom_size), min(image_3d.shape[0], y_CH2 + zoom_size)

                zoomed_image_CH1 = image_orig[y1_CH1:y2_CH1, x1_CH1:x2_CH1]
                zoomed_image_CH2 = image_orig[y1_CH2:y2_CH2, x1_CH2:x2_CH2]
                
                ax_zoom_CH1.clear()
                ax_zoom_CH1.imshow(zoomed_image_CH1, cmap="gray")
                ax_zoom_CH1.set_xticks([])
                ax_zoom_CH1.set_yticks([])
                ax_zoom_CH1.set_title(f"Zoomed In ({y_CH1}, {x_CH1})")
                ax_zoom_CH2.clear()
                ax_zoom_CH2.imshow(zoomed_image_CH2, cmap="gray")
                ax_zoom_CH2.set_xticks([])
                ax_zoom_CH2.set_yticks([])
                ax_zoom_CH2.set_title(f"Zoomed In ({y_CH2}, {x_CH2})")

                tot_intensity_all_frames_CH1 = []
                tot_intensity_all_frames_CH2 = []

                for i in range(len(Frames_data)):

                    # transforms from 2D to 3D
                    if Frames_data[i].ndim == 2:
                        frame_data_3d = np.repeat(Frames_data[i][..., np.newaxis], 3, -1)
                    elif Frames_data[i].ndim==3 and Frames_data[i].shape[2]==3:
                        frame_data_3d = Frames_data[i]

                    frame_data_copy = frame_data_3d.copy()
                    frame_data_copy[mask] = [255, 255, 0]

                    total_intensity_CH1,_ = intensity_in_circle(frame_data_3d, radius, y_CH1, x_CH1)
                    total_intensity_CH2,_ = intensity_in_circle(frame_data_3d, radius, y_CH2, x_CH2)
                    tot_intensity_all_frames_CH1.append(total_intensity_CH1)
                    tot_intensity_all_frames_CH2.append(total_intensity_CH2)
                
                time= np.linspace(0, (len(tot_intensity_all_frames_CH1) - 1) * tpf, len(tot_intensity_all_frames_CH1))

                ax_intensity_CH1.clear()
                ax_intensity_CH1.plot(time, tot_intensity_all_frames_CH1, color='g', label='CH2')
                ax_intensity_CH1.set_title(f"Intensity v Time in Donor Peak {idx}")
                ax_intensity_CH1.set_xlabel('Time (s)')
                ax_intensity_CH1.set_ylabel('Intensity')
                ax_intensity_CH1.set_ylim(-255, max(tot_intensity_all_frames_CH1)+255)
                ax_intensity_CH1.grid()

                ax_intensity_CH2.clear()
                ax_intensity_CH2.plot(time, tot_intensity_all_frames_CH2, color='b', label='CH2')
                ax_intensity_CH2.set_title(f"Intensity v Time in Acceptor Peak {idx}")
                ax_intensity_CH2.set_xlabel('Time (s)')
                ax_intensity_CH2.set_ylabel('Intensity')
                ax_intensity_CH2.set_ylim(-255, max(tot_intensity_all_frames_CH2)+255)
                ax_intensity_CH2.grid()

                FRET_values = calc_FRET(tot_intensity_all_frames_CH1, tot_intensity_all_frames_CH2)
                ax_FRET.clear()               
                ax_FRET.plot(time, FRET_values, color='r')
                ax_FRET.set_title(f"FRET v Time in Pair {idx}")
                ax_FRET.set_xlabel('Time (s)')
                ax_FRET.set_ylabel('FRET Efficiency')
                ax_FRET.grid()

                dist_values = calc_distance(FRET_values, R_0)
                ax_dist.clear()
                ax_dist.plot(time, dist_values, color='y')
                ax_dist.set_title(f"Distance v Time in Pair {idx}")
                ax_dist.set_xlabel('Time (s)')
                ax_dist.set_ylabel('Distance')
                ax_dist.grid()

                rect1 = patches.Rectangle((x1_CH1, y1_CH1), x2_CH1 - x1_CH1, y2_CH1 - y1_CH1, linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect1)
                rect2 = patches.Rectangle((x1_CH2, y1_CH2), x2_CH2 - x1_CH2, y2_CH2 - y1_CH2, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect2)

    annot.set_visible(visible)
    fig.canvas.draw_idle()


def on_hover_intensity_merged(event, pma_file_path, fig, ax, scatter_data, image_3d, image_orig, mask, radius=4, tpf=1/100, R_0=56, Intense_axes=[0.48, 0.6, 0.5, 0.3], FRET_axes=[0.48, 0.35, 0.5, 0.15], dist_axes=[0.48, 0.1, 0.5, 0.15], CH1_zoom_axes=[0.04, 0.06, 0.15, 0.15], CH2_zoom_axes=[0.23, 0.06, 0.15, 0.15]):
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

                y_CH1, x_CH1 = scatter_data[0][1][idx]
                x1_CH1, x2_CH1 = max(0, x_CH1 - zoom_size), min(image_3d.shape[1], x_CH1 + zoom_size+1)
                y1_CH1, y2_CH1 = max(0, y_CH1 - zoom_size), min(image_3d.shape[0], y_CH1 + zoom_size+1)
                y_CH2, x_CH2 = scatter_data[1][1][idx]
                x1_CH2, x2_CH2 = max(0, x_CH2 - zoom_size), min(image_3d.shape[1], x_CH2 + zoom_size+1)
                y1_CH2, y2_CH2 = max(0, y_CH2 - zoom_size), min(image_3d.shape[0], y_CH2 + zoom_size+1)

                zoomed_image_CH1 = image_orig[y1_CH1:y2_CH1, x1_CH1:x2_CH1]
                zoomed_image_CH2 = image_orig[y1_CH2:y2_CH2, x1_CH2:x2_CH2]
                
                ax_zoom_CH1.clear()
                ax_zoom_CH1.imshow(zoomed_image_CH1, cmap="gray")
                ax_zoom_CH1.set_xticks([])
                ax_zoom_CH1.set_yticks([])
                ax_zoom_CH1.set_title(f"Zoomed In ({y_CH1}, {x_CH1})")
                ax_zoom_CH2.clear()
                ax_zoom_CH2.imshow(zoomed_image_CH2, cmap="gray")
                ax_zoom_CH2.set_xticks([])
                ax_zoom_CH2.set_yticks([])
                ax_zoom_CH2.set_title(f"Zoomed In ({y_CH2}, {x_CH2})")

                tot_intensity_all_frames_CH1 = []
                tot_intensity_all_frames_CH2 = []

                for i in range(len(Frames_data)): #for i in range(795): i= 0, 1, 2,..., 794

                    # transforms from 2D to 3D
                    if Frames_data[i].ndim == 2:
                        frame_data_3d = np.repeat(Frames_data[i][..., np.newaxis], 3, -1)
                    elif Frames_data[i].ndim==3 and Frames_data[i].shape[2]==3:
                        frame_data_3d = Frames_data[i]

                    frame_data_copy = frame_data_3d.copy()
                    frame_data_copy[mask] = [255, 255, 0]

                    total_intensity_CH1,_ = intensity_in_circle(frame_data_3d, radius, y_CH1, x_CH1)
                    total_intensity_CH2,_ = intensity_in_circle(frame_data_3d, radius, y_CH2, x_CH2)
                    tot_intensity_all_frames_CH1.append(total_intensity_CH1)
                    tot_intensity_all_frames_CH2.append(total_intensity_CH2)

                time= np.linspace(0, (len(tot_intensity_all_frames_CH1) - 1) * tpf, len(tot_intensity_all_frames_CH1))
                ax_intensity.clear()
                ax_intensity.plot(time, tot_intensity_all_frames_CH1, color='g', label='CH1')
                ax_intensity.plot(time, tot_intensity_all_frames_CH2, color='b', label='CH2')
                ax_intensity.set_title(f"Intensity v Time in Peak {idx}")
                ax_intensity.set_xlabel('Time (s)')
                ax_intensity.set_ylabel('Intensity')
                ax_intensity.set_ylim(-255, max(max(tot_intensity_all_frames_CH1), max(tot_intensity_all_frames_CH2))+255)
                ax_intensity.legend(bbox_to_anchor=(1.0, 1.22), loc='upper right')
                ax_intensity.grid()
                ax_intensity.set_xlim(0, time[-1])

                FRET_values = calc_FRET(tot_intensity_all_frames_CH1, tot_intensity_all_frames_CH2)
                ax_FRET.clear()               
                ax_FRET.plot(time, FRET_values, color='r')
                ax_FRET.set_title(f"FRET v Time in Pair {idx}")
                ax_FRET.set_xlabel('Time (s)')
                ax_FRET.set_ylabel('FRET Efficiency')
                ax_FRET.set_xlim(0, time[-1])
                ax_FRET.grid()

                dist_values = calc_distance(FRET_values, R_0)
                ax_dist.clear()
                ax_dist.plot(time, dist_values, color='y')
                ax_dist.set_title(f"Distance v Time in Pair {idx}")
                ax_dist.set_xlabel('Time (s)')
                ax_dist.set_ylabel('Distance')
                ax_dist.set_xlim(0, time[-1])
                ax_dist.grid()
            
                rect1 = patches.Rectangle((x1_CH1, y1_CH1), x2_CH1 - x1_CH1, y2_CH1 - y1_CH1, linewidth=1, edgecolor='g', facecolor='none')
                ax.add_patch(rect1)
                rect2 = patches.Rectangle((x1_CH2, y1_CH2), x2_CH2 - x1_CH2, y2_CH2 - y1_CH2, linewidth=1, edgecolor='b', facecolor='none')
                ax.add_patch(rect2)

    annot.set_visible(visible)
    fig.canvas.draw_idle()


def count_circle(radius, y_centre=15, x_centre=15):
    # x = radius
    # y = 0
    total = 0
    #filling in the circle
    for i in range(x_centre - radius, x_centre + radius + 1):
        for j in range(y_centre - radius, y_centre + radius + 1):
            if (i - x_centre) ** 2 + (j - y_centre) ** 2 < radius ** 2:
                total +=1
    
    return total

def sgl_frame_intense_arr(input_array, radius, y_centre_arr, x_centre_arr):

    intensity_arr_all_peaks = []
    total_arr = []

    #filling in the circle
    for y_centre, x_centre in zip(y_centre_arr, x_centre_arr):
        total = 0
        for i in range(x_centre - radius, x_centre+ radius + 1):
            for j in range(y_centre - radius, y_centre + radius + 1):
                if (i - x_centre) ** 2 + (j - y_centre) ** 2 < radius ** 2:
                    intensity_arr_all_peaks.append(input_array[j][i][2])
                    total += int(input_array[j][i][2])
        total_arr.append(total)

    return intensity_arr_all_peaks, total_arr

def intensity_in_circle(input_array, radius, y_centre, x_centre):
    total_intensity = 0
    intensity_arr = []
    #filling in the circle
    for i in range(x_centre - radius, x_centre + radius + 1):
        for j in range(y_centre - radius, y_centre + radius + 1):
            if (i - x_centre) ** 2 + (j - y_centre) ** 2 < radius ** 2:
                intensity_arr.append(int(input_array[j][i][2]))
                total_intensity += int(input_array[j][i][2])

    return total_intensity, intensity_arr

def calc_FRET(I_D_list, I_A_list):
    I_D, I_A = np.array(I_D_list), np.array(I_A_list)
    FRET_arr = I_A/(I_D + I_A)
    return FRET_arr.tolist()

def calc_distance(FRET_list, R_0):
    d = R_0 * ((1/np.array(FRET_list)) - 1)**(1/6)
    return d.tolist()


def static_global_background_subtraction(pma_file_path, input_array, radius, y_centre_arr, x_centre_arr):
    frames_data = read_pma(pma_file_path) 
    all_peaks_intensity = 0
    pixel_count = 0
    #filling in the circle
    for y_centre, x_centre in zip(y_centre_arr, x_centre_arr):
        for i in range(x_centre - radius, x_centre+ radius + 1):
            for j in range(y_centre - radius, y_centre + radius + 1):
                if (i - x_centre) ** 2 + (j - y_centre) ** 2 < radius ** 2:
                    all_peaks_intensity += int(input_array[i][j][0])
                    pixel_count += 1
    
    # by summing the third column of the array we exclude the yellow pixels from being included!
    total_intensity = np.sum(input_array[:, :,2])
    
    num_of_peaks = len(y_centre_arr)
    num_of_peak_pixels = count_circle(radius) * num_of_peaks
    num_of_frame_pixels = input_array.shape[0] * input_array.shape[1]

    #avg_peak_intensity gives the avg intensity of the pixels that are not within the yellow circle
    intensity_to_remove = (total_intensity-all_peaks_intensity) // (num_of_frame_pixels-num_of_peak_pixels)
    corrected_frames_data = []
    for frame in frames_data: #frame is 1D
        frame = frame - intensity_to_remove
        corrected_frames_data.append(frame)
    return corrected_frames_data