import cv2
import os
import argparse

"""
Use: python3 imagesToVideo.py <image_folder_path> <video_file_path>
Generates a mp4 video file using the images in <image_folder_path>
"""

def generateVideoFile(image_folder_path, video_file_path):
    # Load and sort the images based on their filenames numerically
    images = [img for img in os.listdir(image_folder_path) if img.endswith(".png")]
    images = sorted(images, key=lambda x: int(x.split('.')[0]))

    # Get the first image to determine the video frame size
    first_image = cv2.imread(os.path.join(image_folder_path, images[0]))
    height, width, layers = first_image.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec (you can change this to other codecs)
    video = cv2.VideoWriter(video_file_path, fourcc, 30, (width, height))  # 30 FPS

    # Iterate through the images, write each frame to the video, and display it
    counter = 0
    for image in images:
        img_path = os.path.join(image_folder_path, image)
        frame = cv2.imread(img_path)
        video.write(frame)
        counter += 1
        print("image #" + str(counter))

    video.release()  # Release the VideoWriter

    print("Video creation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_folder_path', type=str)
    parser.add_argument('video_file_path', type=str)
    args = parser.parse_args()

    generateVideoFile(args.image_folder_path, args.video_file_path)

