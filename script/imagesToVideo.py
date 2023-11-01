import cv2
import os

# 1. Define the folder containing your image sequence and the output video file path
image_folder = '/home/dev_ws/visual_slam/data/image_2/'
video_name = '/home/dev_ws/visual_slam/data/video_2.mp4'

# 2. Load and sort the images based on their filenames numerically
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = sorted(images, key=lambda x: int(x.split('.')[0]))

# 3. Get the first image to determine the video frame size
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

# 4. Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec (you can change this to other codecs)
video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))  # 30 FPS

# 5. Iterate through the images, write each frame to the video, and display it
counter = 0
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)
    counter += 1
    print("image #" + str(counter))

video.release()  # Release the VideoWriter

print("Video creation complete.")

