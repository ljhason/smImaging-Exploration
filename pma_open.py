import sys
import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from scipy.ndimage import maximum_filter, label
from skimage import io, feature, draw
from PIL import Image
from skimage.util.shape import view_as_blocks
from skimage.feature import peak_local_max

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
            Nframes = (filesize - 4) // (X_pixels * Y_pixels)  #Assuming 4-byte header
            f.seek(0, 4) #Reset file pointer to immediately after 4 byte header

            #Read the binary image data
            frame_data0 = f.read(X_pixels * Y_pixels)
            #Convert the frame data into a 2D numpy array of size (Y_pixels, X_pixels)
            image_data = np.frombuffer(frame_data0, dtype=np.uint8).reshape((Y_pixels, X_pixels))

            return image_data

    except Exception as e:
        print(f"Error reading .pma file: {e}")
        return None

def read_pma(pma_file_path):
    try:
        with open(pma_file_path, "rb") as f:
            #Assign X_pixels and Y_pixels as the first two 16-bit integers in the file
            #<:little-endian (least significant byte first), HH:two 16-bit integers
            X_pixels, Y_pixels = struct.unpack("<HH", f.read(4))
            print(f"Image Size: {X_pixels} x {Y_pixels}")
            
            #Calc number of frames
            f.seek(0, 2) #sets pointer to end of file .seek(offset, from_what)
            filesize = f.tell() #returns current (end) position of pointer
            Nframes = (filesize - 4) // (X_pixels * Y_pixels)  #Assuming 4-byte header
            f.seek(0, 4) #Reset file pointer to immediately after 4 byte header

            #Return a list of 2D numpy arrays, each representing a frame
            return [np.frombuffer(f.read(X_pixels * Y_pixels), dtype=np.uint8).reshape((Y_pixels, X_pixels)) for frame_idx in range(Nframes)]

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


#Note to self: This function is NOT returning creating a PNG file!! 
def avg_frame_png(pma_file_path):
    try:
        Frames_data = read_pma(pma_file_path)
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
def good_peak_finder_CH1(image_path, sigma=3, block_size=16, scaler_percent=32, boarder=10, max_rad=3):
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

def good_peak_finder_CH2(image_path, sigma=2, block_size=16, scaler_percent=10, boarder=10, max_rad=3):
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

# make the arrow point from below the point

def init_annot(ax, text="", xy=(0, 0), xytext=(0, 10),textcoords="offset points", bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->")):
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


# Event listener for hover functionality
# Please note that python uses [row,col] however I print [x,y] therefore transformations need to be done and users must be wary of this
def on_event(event, fig, scatter_data):
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

def find_linear_pairs(peaks_1, peaks_2, tolerance=1):
    # peaks_2 coordinates goes from [0, 512] to [256,512]
    gp1_list = [tuple(peak) for peak in peaks_1]
    gp2_list = [tuple(peak) for peak in peaks_2]
    gp2_set = set(gp2_list)
    linear_pair_count = 0
    linear_pair_arr_CH1 = []
    linear_pair_arr_CH2 = []

    for coord in gp1_list:
            for c in gp2_set:
                if (abs(coord[0] - c[0])) <=tolerance and (256-tolerance <= abs(coord[1] - c[1]) <= 256+tolerance) and c not in linear_pair_arr_CH2:
                    linear_pair_count += 1
                    linear_pair_arr_CH1.append(coord)
                    linear_pair_arr_CH2.append(c)
                    break
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

# Some 
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