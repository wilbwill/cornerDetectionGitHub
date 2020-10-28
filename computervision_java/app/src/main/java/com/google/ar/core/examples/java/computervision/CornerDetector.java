package com.google.ar.core.examples.java.computervision;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Point;

import java.nio.ByteBuffer;


public class CornerDetector {
    private byte[] inputPixels = new byte[0]; // Reuse java byte array to avoid multiple allocations.

    public synchronized ByteBuffer detect(int width, int height, int stride, ByteBuffer input) {
        if (stride * height > inputPixels.length) {
            inputPixels = new byte[stride * height];
        }

        byte[] outputPixels = new byte[width * height];
        input.position(0);
        input.get(inputPixels, 0, input.capacity());

        // Convert input byte buffer into opencv Mat
        byte[] data = new byte[input.capacity()];
        ((ByteBuffer) input.duplicate().clear()).get(data);

        Mat scene = new Mat(height, width, CvType.CV_8UC4);
        scene.put(0, 0, data);
        Mat object = new Mat(height, width, CvType.CV_8UC4);
        object.put(0, 0, data);

        Mat outMat = Harris(scene, object, 10000);

        //turn Mat into byte[] into byteBuffer
        byte[] buffer = new byte[(int) (outMat.total() * outMat.channels())];
        outMat.get(0,0,buffer);

        return ByteBuffer.wrap(buffer);
    }

    private Mat Harris(Mat Scene, Mat Object, int thresh) {

        // This function implements the Harris Corner detection. The corners at intensity > thresh
        // are drawn.
        Mat Harris_scene = new Mat();
        Mat Harris_object = new Mat();

        Mat harris_scene_norm = new Mat(), harris_object_norm = new Mat(), harris_scene_scaled = new Mat(), harris_object_scaled = new Mat();
        int blockSize = 9;
        int apertureSize = 5;
        double k = 0.1;
        Imgproc.cornerHarris(Scene, Harris_scene,blockSize, apertureSize,k);
        Imgproc.cornerHarris(Object, Harris_object, blockSize,apertureSize,k);

        Core.normalize(Harris_scene, harris_scene_norm, 0, 255, Core.NORM_MINMAX, CvType.CV_32FC1, new Mat());
        Core.normalize(Harris_object, harris_object_norm, 0, 255, Core.NORM_MINMAX, CvType.CV_32FC1, new Mat());

        Core.convertScaleAbs(harris_scene_norm, harris_scene_scaled);
        Core.convertScaleAbs(harris_object_norm, harris_object_scaled);

        for( int j = 0; j < harris_scene_norm.rows() ; j++){
            for( int i = 0; i < harris_scene_norm.cols(); i++){
                if ((int) harris_scene_norm.get(j,i)[0] > thresh){
                    Imgproc.circle(harris_scene_scaled, new Point(i,j), 5 , new Scalar(0), 2 ,8 , 0);
                }
            }
        }

        for( int j = 0; j < harris_object_norm.rows() ; j++){
            for( int i = 0; i < harris_object_norm.cols(); i++){
                if ((int) harris_object_norm.get(j,i)[0] > thresh){
                    Imgproc.circle(harris_object_scaled, new Point(i,j), 5 , new Scalar(0), 2 ,8 , 0);
                }
            }
        }

        return harris_scene_scaled;
    }

}
