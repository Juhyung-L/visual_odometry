# visual_odometry
Visual odometry using OpenCV
SIFT features are extracted from each frame of the KITTI dataset video. Features from subsequent frames are matched to estimate the change in pose of the camera. This is just a raw visual odometry without any error correction techniques, such as loop closure and pose graph optimization. Hence, there is severe odometric drift.

Detailed explanation at: https://juhyunglee0313.wixsite.com/portfolio/post/monocular-visual-odometry

Results:

![Screenshot from 2024-01-10 14-53-37](https://github.com/Juhyung-L/visual_odometry/assets/102873080/523ff3e0-3fcf-48b9-b24d-1533a0d06378)

Graph comparison between ground-truth pose and estimated pose.

![Screenshot from 2024-01-10 14-52-40](https://github.com/Juhyung-L/visual_odometry/assets/102873080/fff83107-9d63-41b0-a5b5-25e0281250a8)


Footages:
[![Watch the video](https://img.youtube.com/vi/pQ5rXVprvJE/maxresdefault.jpg)](https://www.youtube.com/watch?v=pQ5rXVprvJE "Visual Odometry (KITTI Dataset)")
