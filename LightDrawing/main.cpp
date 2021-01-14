#include <iostream>
#include "opencv2/opencv.hpp"
#include "MojeCV.hpp"

using namespace std;
using namespace cv;


Mat dil(Mat src)
{
    Mat a = src.clone();
    Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    Mat dst;
    dilate(a,dst,kernel);
    return dst;
}

Mat ero(Mat src)
{
    Mat a = src.clone();
    Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
    Mat dst;
    erode(a,dst,kernel);
    return dst;
}

int main( int argc, char *argv[] )
{
    Mat frame;
    Mat bw,gray;
    Mat image(320, 240, CV_8UC3, cv::Scalar(0, 0, 0)); // kreslici platno
    Mat mem;

    string filename = "";
    string fileOut = "";
    int START = 0;
    int LEN = 1;
    if (argc > 4)
    {
        string arg1(argv[1]);
        string arg2(argv[2]);
        string arg3(argv[3]);
        string arg4(argv[4]);
        filename = arg1;
        fileOut = arg2;
        START = stoi(arg3);
        LEN = stoi(arg4);
    }
    else
    {
        cout << "arg1:input arg2:output arg3:start arg4:len" << endl;
        return 0;
    }

    VideoCapture cap(filename);
    if(!cap.isOpened())
        return -1;

    // set frame interval
    if(START > 0)
    {
        cap.set(CAP_PROP_POS_FRAMES,START);
    }else
    {
        cap.set(CAP_PROP_POS_FRAMES,0);
    }
    if(LEN == 0)
    {
        LEN = cap.get(CAP_PROP_FRAME_COUNT);
    }

    cap >> frame;
    int W = frame.cols;
    int H = frame.rows;
    int framerate = cap.get(CAP_PROP_FPS);
    //int frame_count = cap.get(CAP_PROP_FRAME_COUNT);
    int frame_count = LEN;
    int iterace = 0;
    resize(image,image,Size(W,H));
    mem = frame.clone();

    // Save video
    VideoWriter video(fileOut,VideoWriter::fourcc('p','n','g',' '),framerate, Size(W,H));

    for(;;)
    {

        cap >> frame; // obnova framebufferu
        if(frame.empty())
        {
            return 1;
        }

        // #1 treshold
        cvtColor(frame,gray,COLOR_BGR2GRAY);
        threshold(gray, bw, 240, 255, 0); // 240 default
        cvtColor(bw,bw,COLOR_GRAY2BGR);



        bw = dil(bw);
        blur(bw,bw,Size(5,5));
        image = MOJECV::lighten(image,bw);
        mem = MOJECV::lighten(mem,frame);
        Mat test = MOJECV::AlphaBlend(mem,frame,image,Point(0,0)); // výstupní produkt
        Mat dst = MOJECV::Mix(frame,test,0.1); // mix s původním framem


        MOJECV::preview nahled;
        nahled.frame = dst;
        nahled.iterace = iterace;
        nahled.frame_count = frame_count;
        imshow("nahled",nahled.nahled());

        iterace++;
        video.write(dst);
        if (iterace == LEN)
        {
            break;
        }
        char key = (char)waitKeyEx(20);
        if( key == 27 || key == 'q' || key == 'Q' ) // 'ESC'
            break;
    }
    return 0;
}
