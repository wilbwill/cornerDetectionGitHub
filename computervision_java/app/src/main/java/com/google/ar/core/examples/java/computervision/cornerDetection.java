package com.google.ar.core.examples.java.computervision;

import android.media.Image;

import com.quickbirdstudios.yuv2mat.Yuv;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;

public class cornerDetection {

    public synchronized ByteBuffer detect(int width, int height, int stride, Image input) {
        System.loadLibrary("opencv_java4");

        // convert from yuv image to rgb mat, then to grayscale
        Mat mat = Yuv.rgb(input);

        Mat matGray = new Mat();
        Imgproc.cvtColor(mat, matGray, Imgproc.COLOR_RGB2GRAY);

        Mat outMat = Harris(matGray, 10000);

        // Convert back to rgb to produce Y bytes
        Mat rgbMat = new Mat();
        Imgproc.cvtColor(outMat, rgbMat, Imgproc.COLOR_GRAY2RGB);

        byte[] retBuffer = new byte[(width * height)];
        // Convert values from rgbMat to Y bytes


        // Method 1 variables:
        int max_row = -1;
        int min_row = height;

        // Method 2 variables:
        /*
        int max_row = -1;
        int min_row = height;
        int max_counter = 0;
        int min_counter = 0;
         */

        for(int i = 0; i < height; ++i) {
            // Method 2 variable:
            // int current_sum = 0;
            for(int j = 0; j < width; ++j) {
                double[] rgb = rgbMat.get(i, j);
                double R = rgb[0];
                double G = rgb[1];
                double B = rgb[2];
                int y = (int) ((0.299 * R) + (0.587 * G) + (0.114 * B));
                if(y < 125) y = 0;
                retBuffer[i * width + j] = (byte) y;

                //  METHOD 1: SET VALUES IN SAME ROW TO 255
                if (y > 200){
                    if (i > max_row) max_row = i;
                    if (i < min_row) min_row = i;
                }

                //  METHOD 2: COUNT VALUES IN SAME ROW
                /*
                if (y > 200){
                   current_sum += 1;
                }
                 */
            }
            // METHOD 2: COUNT VALUES IN SAME ROW
            /*
            if (current_sum > max_counter && i > max_row){
                max_counter = current_sum;
                max_row = i;
             }
             else if (current_sum > min_counter && i < min_row){
                min_counter = current_sum;
                min_row = i;
             }
             */
        }

        //          USED FOR BOTH METHOD 1 AND 2
        for (int j = 0; j < width; j++){
            retBuffer[max_row * width + j] = (byte)255;
            retBuffer[min_row * width + j] = (byte)255;
        }

        return ByteBuffer.wrap(retBuffer);
    }

    private Mat Harris(Mat Object, int thresh) {

        // This func implements the Harris Corner detection. The corners at intensity > thresh
        // are drawn.

        Mat Harris_object = new Mat();

        Mat harris_object_norm = new Mat(), harris_object_scaled = new Mat();
        int blockSize = 9;
        int apertureSize = 3;
        double k = 0.1;
        Imgproc.cornerHarris(Object, Harris_object, blockSize,apertureSize,k);

        Core.normalize(Harris_object, harris_object_norm, 0, 255, Core.NORM_MINMAX, CvType.CV_32FC1, new Mat());

        Core.convertScaleAbs(harris_object_norm, harris_object_scaled);

        for( int j = 0; j < harris_object_norm.rows() ; j++){
            for( int i = 0; i < harris_object_norm.cols(); i++){
                if ((int) harris_object_norm.get(j,i)[0] > thresh){
                    Imgproc.circle(harris_object_scaled, new Point(i,j), 5 , new Scalar(0), 2 ,8 , 0);
                }
            }
        }

        return harris_object_scaled;
    }


}
