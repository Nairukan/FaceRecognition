#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "myfacerecognition.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>










/*void Vizualizate(Mat image){
    auto w=image.cols;
    unsigned m[image.rows*image.cols];
    for (int i=0; i<image.rows; i++){
        for (int j=0; j<image.cols; j++){
            m[i*w+j]=image.at<unsigned>(Point(i,j));
        }
    }
    for (uint i=1; i<image.rows-1; i++){
        for (uint j=1; j<image.cols-1; j++){
            unsigned center=m[i*w+j];
            for (uint ii=-1; ii<2; ii++){
                for (uint ij=-1; ij<2; ij++){
                    if (m[(i+ii)*w+j+ij]>=center) m[(i+ii)*w+j+ij]=1;
                    else m[(i+ii)*w+j+ij]=0;
                }
            }
            m[i*w+j]=center;
        }
    }
    imshow("adsada", image);
    //waitKey(300000);
}*/

/*inline Mat GetNormalizateImage(Mat image, Mat& output){
    image.copyTo(output);
    //resize(answer, answer, Size(300, 300));
    //GaussianBlur(answer, answer, Size(5, 5), 0);
    //medianBlur(answer, answer, 5);
    Mat chanels[image.channels()];
    split(output, chanels);
    for (auto now : chanels){
        auto CLAHE = createCLAHE(3, Size(5,5));
        CLAHE->apply(now, now);
    }
    merge(chanels, image.channels(), output);
}
*/

void CelebrityTest(){
    vector<int> Test;
    {
        MyFaceRecognition FR;
    ///    FR.PrepareDataset("/home/nikita/QtProj/build-Some-Desktop-Debug/Dataset.csv");
    }
    MyFaceRecognition FR;
    FR.Init("/home/nikita/QtProj/build-LBPH-Desktop-Debug/Norm_Face/Faces.csv", "LBPH.yaml");
    cout << "PREPARE_END\n";
    vector<int> labels=FR.model->getLabels();
    vector<Mat> TestMat;
    ifstream itest("/home/nikita/QtProj/build-Some-Desktop-Debug/Test.csv");
    int countIdiotError=0;
    int countOK_LPBH=0;
    int countBAD_LPBH=0;
    while(!itest.eof()){
        string path; itest >> path;
        if (path!=""){
            TestMat.push_back(imread(path));
            int numo; itest >> numo;
            Test.push_back(numo/1000);
        }
    }
    int n=Test.size();
    for (int i=0; i<Test.size(); i++){
        if (FR.ExtractFace(TestMat[i], TestMat[i])){
            //FR.NormalizeImage(TestMat[i], TestMat[i]);
            //FR.some(TestMat[i], TestMat[i]);
            auto res=FR.GetSimilarFacesLBPH(TestMat[i], 21, 30);
            int q=0;
            cout << i << "/" << Test.size() << ": \n";
            for (; q<res.size(); q++){
                if (res[q].first/1000==Test[i]){
                    cout << "OK in stage LBPH\n";
                    countOK_LPBH++;
                    break;
                }
            }
            if (q==res.size()){
                if (find(labels.begin(), labels.end(), Test[i])==labels.end()){
                    n--;
                }
                else{
                    cout << "FAIL in stage LBPH\n";
                    countBAD_LPBH++;
                }
            }
        }else{
            cout << "FAIL in stage Extract\n";
            countIdiotError++;
        }
    }
    cout << "\nOK_LPBH " << double(countOK_LPBH)/n << "\n";
    cout << "BAD_LPBH " << double(countBAD_LPBH)/n << "\n";
    cout << "STUPID_ERROR " << double(countIdiotError)/n << "\n";
}




int main(int argc, const char *argv[]) {
    //CelebrityTest();
    //return 0;
    MyFaceRecognition FR;
    //FR.PrepareDataset("/home/nikita/QtProj/build-OpenCV_tests-Desktop-Debug/LBPH_people.csv");
    //return 0;
    if (argc != 2) {
        cout << "usage: " << argv[0] << " <csv.ext>" << endl;
        exit(1);
    }
    if (!FR.Init(argv[1], "LBPH.yaml")) return 0;
    VideoCapture capture(0);
    Mat ProcessingImage, Im;
    long st;
    uint n=36;
    while (true){
        n++;
        st=clock();



        {
            ProcessingImage=imread("MyWeb_"+to_string(n)+".jpg", IMREAD_GRAYSCALE);
            Mat descriptor;
            //auto CLAHE = createCLAHE(40, Size(6,6));
            //CLAHE->apply(ProcessingImage, ProcessingImage);
            cv::FileStorage fs2("MyWebDescript_"+to_string(n)+".yaml", cv::FileStorage::READ);
            fs2["descriptors"] >> descriptor;
            fs2.release();
            Mat d2=FR.getDescriptByLabel(65782);
            Mat im2=FR.getMatByLabel(65782);
            auto Matches=FR._compareFacesWithSIFT(descriptor, d2);

            vector<KeyPoint> kp1, kp2=FR.getKeyPointByLabel(65782);


            cv::FileStorage fsk1("MyWebKeyPoints_"+to_string(n)+".yaml", cv::FileStorage::READ);
            fsk1["keypoints"] >> kp1;
            fsk1.release();
            Mat out;


            std::vector<cv::Point2f> matchedPointsReference;
            std::vector<cv::Point2f> matchedPointsTest;
            for (const cv::DMatch& match : Matches) {
                matchedPointsReference.push_back(kp1[match.queryIdx].pt);
                matchedPointsTest.push_back(kp2[match.trainIdx].pt);
            }
            cv::Mat homography = cv::findHomography(matchedPointsReference, matchedPointsTest, cv::RANSAC);
            std::vector<cv::DMatch> filteredMatches;
            long double sum_mist=0.0;
            for (size_t i = 0; i < Matches.size(); ++i) {
                cv::Mat pointReference = (cv::Mat_<double>(3, 1) << matchedPointsReference[i].x, matchedPointsReference[i].y, 1.0);
                cv::Mat transformedPoint = homography * pointReference;
                double dx = transformedPoint.at<double>(0, 0) - matchedPointsTest[i].x;
                double dy = transformedPoint.at<double>(1, 0) - matchedPointsTest[i].y;
                double distance = std::sqrt(dx*dx + dy*dy);
                if (distance < 100 && Matches[i].distance<80) {
                    filteredMatches.push_back(Matches[i]);
                    sum_mist+=Matches[i].distance;
                }
            }
            Matches=filteredMatches;
            sum_mist/=Matches.size();


            drawMatches(ProcessingImage, kp1, im2, kp2, Matches, out);
            imwrite(to_string(n)+"_SIFT_DELTAS_"+to_string(Matches.size())+".jpg", out);
        }
        continue;



        capture >> Im;

        if (Im.empty()) break;
        Im.copyTo(ProcessingImage);
        imshow("Input", ProcessingImage);

        if (FR.ExtractFace(Im, ProcessingImage)){
            auto CLAHE = createCLAHE(10);
            CLAHE->apply(ProcessingImage, ProcessingImage);
            auto res=FR.GetSimilarFacesLBPH(ProcessingImage, 21, 30, n);
            imshow("Processing", ProcessingImage);
            //putText(ProcessingImage, )
            Mat Result;
            vector<uint> inds;
            for (int i=0; i<res.size(); i++){
                if (Result.empty()) Result=FR.getMatByLabel(res[i].first);
                else{

                    hconcat(Result, FR.getMatByLabel(res[i].first), Result);
                }
                if (res[i].first==65782 || res[i].first==3 || res[i].first==1 || res[i].first==2 || res[i].first==5 || res[i].first==4)
                    inds.push_back(i);
            }
            if (!Result.empty())
                cvtColor(Result, Result, COLOR_GRAY2BGR);
            for (auto now: inds){
                rectangle(Result, Point(512*now,0), Point(512*(now+1)-2, 510), Scalar(0,255,0), 4);
            }
            if (!Result.empty()){
                resize(Result, Result, Size(256*min<int>(7, res.size()), 256));
                imshow("Result", Result);
            }


        }

        //cout << (clock()-st)/CLOCKS_PER_SEC << "s\n\n";
        if (waitKey(max<int>(1,30-(clock()-st)/CLOCKS_PER_SEC))==27) break;
    }


    /*cout << "Init...\n";

    if (!Init(argv[1])) return 0;
    cout << "Start processing.\n";
    Processing(1);
    */
    FR.model->write("LBPH.yaml");

    return 0;
}
