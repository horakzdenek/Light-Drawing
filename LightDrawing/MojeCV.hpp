// nadstavba pro OpenCV verzi 4.4.0
// update: 2020-10-20
#ifndef MOJECV_HPP_INCLUDED
#define MOJECV_HPP_INCLUDED

#include <iostream>
#include <opencv2/opencv.hpp>



using namespace std;
using namespace cv;



namespace MOJECV
{
    class preview
    {
        public:
            Mat frame;
            int iterace = 0;
            int frame_count = 1;
            string hint = "";

            Mat nahled()
            {
                Mat tmp;
                double pomer = (double)frame.cols/(double)frame.rows;
                resize(frame,tmp, Size((int)round(300*pomer),300));
                double pct = (double)iterace/(double)frame_count*100;
                int p = (int)round(pct);
                Point pt(0,12);
                Point pt1(0,0);
                Point pt2((int)round(0.01*p*tmp.cols),13);
                rectangle(tmp, pt2, pt1, Scalar(0, 0, 255), -1, 8, 0); /// Scalar definuje barvu -1 je výplň/tloušťka

                String text = to_string(p)+"%";
                if (iterace == 0)
                {
                    text = "";
                }
                int fontFace = FONT_HERSHEY_SIMPLEX;
                double fontScale = 0.4;
                int thickness = 1;
                putText(tmp, text, pt, fontFace, fontScale, Scalar(0, 255, 0), thickness, 8);
                if (!(hint == ""))
                {
                    rectangle(tmp, Point(tmp.cols,tmp.rows), Point(0,tmp.rows-15), Scalar(120, 120, 120), -1, 8, 0);
                    putText(tmp, hint, Point(0,tmp.rows-4), fontFace, fontScale, Scalar(0, 255, 0), thickness, 8);
                }
                return tmp;
            }

    };

Mat blend_multiply(const Mat& level1, const Mat& level2, uchar opacity)
{
    CV_Assert(level1.size() == level2.size());
    CV_Assert(level1.type() == level2.type());
    CV_Assert(level1.channels() == level2.channels());

    // Get 4 channel float images
    Mat4f src1, src2;

    if (level1.channels() == 3)
    {
        Mat4b tmp1, tmp2;
        cvtColor(level1, tmp1, COLOR_BGR2BGRA);
        cvtColor(level2, tmp2, COLOR_BGR2BGRA);
        tmp1.convertTo(src1, CV_32F, 1. / 255.);
        tmp2.convertTo(src2, CV_32F, 1. / 255.);
    }
    else
    {
        level1.convertTo(src1, CV_32F, 1. / 255.);
        level2.convertTo(src2, CV_32F, 1. / 255.);
    }

    Mat4f dst(src1.rows, src1.cols, Vec4f(0., 0., 0., 0.));

    // Loop on every pixel

    float fopacity = opacity / 255.f;
    float comp_alpha, new_alpha;

    for (int r = 0; r < src1.rows; ++r)
    {
        for (int c = 0; c < src2.cols; ++c)
        {
            const Vec4f& v1 = src1(r, c);
            const Vec4f& v2 = src2(r, c);
            Vec4f& out = dst(r, c);

            comp_alpha = min(v1[3], v2[3]) * fopacity;
            new_alpha = v1[3] + (1.f - v1[3]) * comp_alpha;

            if ((comp_alpha > 0.) && (new_alpha > 0.))
            {
                float ratio = comp_alpha / new_alpha;

                out[0] = max(0.f, min(v1[0] * v2[0], 1.f)) * ratio + (v1[0] * (1.f - ratio));
                out[1] = max(0.f, min(v1[1] * v2[1], 1.f)) * ratio + (v1[1] * (1.f - ratio));
                out[2] = max(0.f, min(v1[2] * v2[2], 1.f)) * ratio + (v1[2] * (1.f - ratio));
            }
            else
            {
                out[0] = v1[0];
                out[1] = v1[1];
                out[2] = v1[2];
            }

            out[3] = v1[3];

        }
    }

    Mat3b dst3b;
    Mat4b dst4b;
    dst.convertTo(dst4b, CV_8U, 255.);
    cvtColor(dst4b, dst3b, COLOR_BGRA2BGR);

    return dst3b;
}

    Mat Mix(Mat A, Mat B, double mix)
    {
        Mat dst;
        Mat a = A.clone();
        Mat b = B.clone();
        double beta = ( 1.0 - mix );
        addWeighted( A, mix, B, beta, 0.0, dst);
        return dst;
    }

    Mat Wave(Mat src, double period, double phase, int amp)
    {
        Mat result = src.clone();
        Vec3b color_dest;
        Point position = Point(0,0);

        for(int y = 0; y < src.rows; y++)
            {
                for (int x = 0; x < src.cols; x++)
                {
                    position.x = (int)(x + sin(y*period+phase)*amp);
                    position.y = (int)(y + sin(x*period+phase)*amp);

                    Vec3b intensity = src.at<Vec3b>(position);
                    int blue = intensity.val[0];
                    int green = intensity.val[1];
                    int red = intensity.val[2];

                    color_dest.val[0] = blue;
                    color_dest.val[1] = green;
                    color_dest.val[2] = red;
                    result.at<Vec3b>(Point(x,y)) = color_dest;

                }
            }
            return result;
    }

    Mat lighten(Mat a, Mat b)
    {
        Mat result = a.clone();
        Vec3b color_dest;
        for(int y = 0; y < a.rows; y++)
        {
            for (int x = 0; x < a.cols; x++)
            {
                // do it
                Vec3b intensity_a = a.at<Vec3b>(y, x);
                Vec3b intensity_b = b.at<Vec3b>(y, x);
                int blue_a = intensity_a.val[0];
                int green_a = intensity_a.val[1];
                int red_a = intensity_a.val[2];

                int blue_b = intensity_b.val[0];
                int green_b = intensity_b.val[1];
                int red_b = intensity_b.val[2];

                if((blue_a + green_a + red_a) > (blue_b + green_b + red_b))
                {
                    color_dest.val[0] = blue_a;
                    color_dest.val[1] = green_a;
                    color_dest.val[2] = red_a;
                }else
                {
                    color_dest.val[0] = blue_b;
                    color_dest.val[1] = green_b;
                    color_dest.val[2] = red_b;
                }

                result.at<Vec3b>(Point(x,y)) = color_dest;

            }
        }

        return result;
    }

    Mat AlphaBlend(Mat foreground, Mat background, Mat mask, Point position)
    {
        Mat a = foreground.clone();
        Mat b = background.clone();

        Mat img(background.size(), CV_8UC3, Scalar(0,0,0));  // foreground
        Mat maska(background.size(), CV_8UC3, Scalar(0,0,0)); // mask

        // insert a pozicování alfa obrázku
        mask.copyTo(maska(Rect(position.x,position.y,mask.cols,mask.rows)));
        a.copyTo(img(Rect(position.x,position.y,a.cols,a.rows)));

        // datové konverze
        img.convertTo(img,CV_32FC3);
        b.convertTo(b,CV_32FC3);
        maska.convertTo(maska,CV_32FC3,1.0/255);

        Mat ouImage = Mat::zeros(img.size(), img.type());
        multiply(maska, img, img);
        multiply(Scalar::all(1.0)-maska, b, b);
        add(img, b, ouImage);

        Mat x = ouImage;
        x.convertTo(x,CV_8UC3);
        return x;
}

    Mat mapping(Mat src, vector<Point2f> inputQuad, vector<Point2f> outputQuad )
    {
        Mat input = src.clone();
        Mat output;
        Mat lambda( 2, 4, CV_32FC1 );
        lambda = Mat::zeros( input.rows, input.cols, input.type() );
        lambda = getPerspectiveTransform( inputQuad, outputQuad );
        warpPerspective(input,output,lambda,output.size() );
        return output;
    }

    Mat AutoAlign(Mat src, Mat sablona, int number_of_iterations)
    {
        Mat im1 = sablona.clone();
        Mat im2 = src.clone();

        Mat im1_gray, im2_gray;
        cvtColor(im1, im1_gray, COLOR_BGR2GRAY);
        cvtColor(im2, im2_gray, COLOR_BGR2GRAY);

        const int warp_mode = MOTION_EUCLIDEAN;
        Mat warp_matrix;

        if ( warp_mode == MOTION_HOMOGRAPHY )
            warp_matrix = Mat::eye(3, 3, CV_32F);
        else
            warp_matrix = Mat::eye(2, 3, CV_32F);

        // Specify the threshold of the increment
        // in the correlation coefficient between two iterations
        double termination_eps = 1e-10;

        // Define termination criteria
        TermCriteria criteria (TermCriteria::COUNT+TermCriteria::EPS, number_of_iterations, termination_eps);

        // Run the ECC algorithm. The results are stored in warp_matrix.
        findTransformECC(
                        im1_gray,
                        im2_gray,
                        warp_matrix,
                        warp_mode,
                        criteria
                    );

        Mat im2_aligned;
        if (warp_mode != MOTION_HOMOGRAPHY)
            // Use warpAffine for Translation, Euclidean and Affine
            warpAffine(im2, im2_aligned, warp_matrix, im1.size(), INTER_LINEAR + WARP_INVERSE_MAP);
        else
            // Use warpPerspective for Homography
            warpPerspective (im2, im2_aligned, warp_matrix, im1.size(),INTER_LINEAR + WARP_INVERSE_MAP);

        return im2_aligned;
    }


    string type2str(int type)
    {
        string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch ( depth )
        {
            case CV_8U:  r = "8U"; break;
            case CV_8S:  r = "8S"; break;
            case CV_16U: r = "16U"; break;
            case CV_16S: r = "16S"; break;
            case CV_32S: r = "32S"; break;
            case CV_32F: r = "32F"; break;
            case CV_64F: r = "64F"; break;
            default:     r = "User"; break;
        }

        r += "C";
        r += (chans+'0');

        return r;
    }

}






#endif // MOJECV_HPP_INCLUDED
