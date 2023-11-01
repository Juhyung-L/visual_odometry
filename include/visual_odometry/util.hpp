#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Vec3d rot2euler(const Matx33d & R)
{
    Mat rotationMatrix(3, 3, CV_64F);
    for (int i=0; i<R.rows; ++i)
    {
        for (int j=0; j<R.cols; ++j)
        {
            rotationMatrix.at<double>(i,j) = R(i,j);
        }
    }

    Vec3d euler;

    double m00 = rotationMatrix.at<double>(0, 0);
    double m02 = rotationMatrix.at<double>(0, 2);
    double m10 = rotationMatrix.at<double>(1, 0);
    double m11 = rotationMatrix.at<double>(1, 1);
    double m12 = rotationMatrix.at<double>(1, 2);
    double m20 = rotationMatrix.at<double>(2, 0);
    double m22 = rotationMatrix.at<double>(2, 2);

    double x, y, z;

    // Assuming the angles are in radians.
    if (m10 > 0.998) { // singularity at north pole
        x = 0;
        y = CV_PI / 2;
        z = atan2(m02, m22);
    }
    else if (m10 < -0.998) { // singularity at south pole
        x = 0;
        y = -CV_PI / 2;
        z = atan2(m02, m22);
    }
    else
    {
        x = atan2(-m12, m11);
        y = asin(m10);
        z = atan2(-m20, m00);
    }

    euler[0] = x * 180.0 / CV_PI;
    euler[1] = y * 180.0 / CV_PI;
    euler[2] = z * 180.0 / CV_PI;

    return euler;
}