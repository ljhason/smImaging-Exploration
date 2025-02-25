import sys
import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image

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