package com.dannymanastireanu.nncamera;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Objects;

public class HaarCascadeView extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    private Button switchButton, btnTakePic;
    private ImageView ivLastImage;
    private CameraBridgeViewBase cameraBridgeViewBase;
    private BaseLoaderCallback baseLoaderCallback;
    private Mat frame, gray, mRgbaF, mRgbaT;

    private Scalar greenColor;
    private MatOfRect matOfRect;
    private int absSignSize;
    private String[] xmlFile = {"stop_sign", "yield", "right_sign", "roundabout", "left_sign", "cars", "bus_front", "speed_limit", "pedestrian", "traffic_light"};
    private CascadeClassifier[] cascadeClassifier;
    private int nextXml = 0;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_haar_cascade_view);

        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                switch (status) {
                    case LoaderCallbackInterface.SUCCESS:
                    {
                        initializeOpenCVDependecies();
                        Log.i("OpenCV", "OpenCV loaded successfully");
                        cameraBridgeViewBase.enableView();
                    } break;
                    default:
                    {
                        super.onManagerConnected(status);
                    } break;
                }
            }
        };

        switchButton = findViewById(R.id.btn_switch);
        btnTakePic = findViewById(R.id.btn_take_pic);
        ivLastImage = findViewById(R.id.img_sign);
        switchButton.setText(xmlFile[nextXml]);
        cameraBridgeViewBase = findViewById(R.id.cameraViewHaar);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
        cameraBridgeViewBase.setMaxFrameSize(720, 1280);
    }

    private void initializeOpenCVDependecies() {

        InputStream[] is = new InputStream[xmlFile.length];
        File[] cascadeDir = new File[xmlFile.length];
        File[] mCascadeFile = new File[xmlFile.length];
        FileOutputStream[] os = new FileOutputStream[xmlFile.length];
        cascadeClassifier  = new CascadeClassifier[xmlFile.length];
        int bytesRead;

        try {
            for (int i = 0; i < xmlFile.length; ++i) {
                is[i] = getResources().openRawResource(getResources().getIdentifier(xmlFile[i], "raw", getPackageName()));
                cascadeDir[i] = getDir(xmlFile[i], Context.MODE_PRIVATE);
                mCascadeFile[i] = new File(cascadeDir[i], xmlFile[i] + ".xml");
                os[i] = new FileOutputStream(mCascadeFile[i]);
                byte[] buffer = new byte[4096];
                while((bytesRead = is[i].read(buffer)) != -1)
                    os[i].write(buffer, 0, bytesRead);
                is[i].close();
                os[i].close();

                cascadeClassifier[i] = new CascadeClassifier(mCascadeFile[i].getAbsolutePath());
            }
        } catch (Exception e) {
            e.getMessage();
        }

    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        gray = new Mat(height, width, CvType.CV_8UC4);
        absSignSize = (int)(height * 0.09);
        greenColor = new Scalar(0, 255, 0);
        matOfRect = new MatOfRect();
        frame = new Mat(height, width, CvType.CV_8UC4);
        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
        mRgbaT = new Mat(height, width, CvType.CV_8UC4);

    }

    @Override
    protected void onResume() {
        super.onResume();
        if(!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, baseLoaderCallback);
        } else {
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onCameraViewStopped() {
        if(cameraBridgeViewBase != null)
            cameraBridgeViewBase.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if(cameraBridgeViewBase != null)
            cameraBridgeViewBase.disableView();
    }

    @Override
    public Mat onCameraFrame(final CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        frame = inputFrame.rgba();

        //rotate camera
        Core.transpose(frame, mRgbaT);
        Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size(), 0,0, 0);
        Core.flip(mRgbaF, frame, 1 );
        Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

        switchButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(nextXml != xmlFile.length - 1)
                    ++nextXml;
                else
                    nextXml = 0;
                switchButton.setText(xmlFile[nextXml]);
                ivLastImage.setImageResource(getResources().getIdentifier(xmlFile[nextXml], "drawable", getPackageName()));

            }
        });

        if(cascadeClassifier[nextXml] != null) {
            cascadeClassifier[nextXml].detectMultiScale(gray, matOfRect, 1.1, 3, 3, new Size(absSignSize, absSignSize), new Size());
        }

        Rect[] elementArray = matOfRect.toArray();
        for (Rect rect : elementArray) {
            Imgproc.rectangle(frame, rect.tl(), rect.br(), greenColor, 3);
        }

        btnTakePic.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");
                String currentDateAndTime = sdf.format(new Date());
                Toast.makeText(getApplicationContext(), "saved", Toast.LENGTH_SHORT).show();
                String filename = "/storage/emulated/0/Signs/samplepass" + currentDateAndTime + ".jpg";
                Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB);
                Imgcodecs.imwrite(filename, frame);
            }
        });

        return frame;
    }
}
