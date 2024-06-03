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
#include <future>
#include <iomanip>










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


int MSER_delta=5, MSER_min_area=60, MSER_max_area=14400;
int MSER_max_variation=25, MSER_min_diversity=20;
int MSER_max_evolution=200;
int MSER_area_treshhold=101, MSER_min_marign=3;
//double MSER_area_treshhold=1.01, MSER_min_marign=0.0030000001;
int MSER_edge_blur_stage=5;
int num=0;
Mat NORAMLS[2151];
Ptr<MSER> mser;

static void on_trackbar(int, void*){

    Mat temp; NORAMLS[num].copyTo(temp);

    //imshow("MSER TEST", temp);

    mser = MSER::create(MSER_delta, MSER_min_area, MSER_max_area, MSER_max_variation/100.0, MSER_min_diversity/100.0, MSER_max_evolution, MSER_area_treshhold/100.0, MSER_min_marign/1000.0, MSER_edge_blur_stage);
     vector<vector<Point>> regions;
     vector<Rect> mserBoundBoxes;
     mser->detectRegions(NORAMLS[num], regions, mserBoundBoxes);
     for (const auto& region : regions){
         vector<Point> contour(region); // Convert MSER region to contour
                 vector<vector<Point>> contours;
                 contours.push_back(contour);
                 drawContours(temp, contours, 0, Scalar(0), 2); // Green color, thickness = 2
     }
     imshow("MSER TEST", temp);

}

static void on_button(int, void (*f)(int, void*)){
    num=(num+1)%2151;
    f(0, 0);
}

void TestMSER(){
    MSER_delta=5; MSER_min_area=200; MSER_max_area=3000;
    MSER_max_variation=80; MSER_min_diversity=10;
    MSER_area_treshhold=200; MSER_edge_blur_stage=18;
    MSER_max_evolution=1000; MSER_min_marign=10;
    mser = MSER::create(MSER_delta, MSER_min_area, MSER_max_area, MSER_max_variation/100.0, MSER_min_diversity/100.0, MSER_max_evolution, MSER_area_treshhold/100.0, MSER_min_marign/1000.0, MSER_edge_blur_stage);
    string temp;
    ifstream file("Norm_Face/Normals.txt");
    for (int i=0; i<2151; ++i){
        getline(file, temp);
        NORAMLS[i]=imread("Norm_Face/"+temp, IMREAD_GRAYSCALE);
        auto CLAHE = createCLAHE(10);
        CLAHE->apply(NORAMLS[i], NORAMLS[i]);
    }
    namedWindow("MSER TEST");
    createTrackbar("MSER_delta", "MSER TEST",           &MSER_delta, 100, on_trackbar);
    createTrackbar("MSER_min_area", "MSER TEST",        &MSER_min_area, 500, on_trackbar);
    createTrackbar("MSER_max_area", "MSER TEST",        &MSER_max_area, 5000, on_trackbar);
    createTrackbar("MSER_max_variation", "MSER TEST",   &MSER_max_variation, 200, on_trackbar);
    createTrackbar("MSER_min_diversity", "MSER TEST",   &MSER_min_diversity, 200, on_trackbar);

    createTrackbar("MSER_max_evolution", "MSER TEST",   &MSER_max_evolution, 5000, on_trackbar);
    createTrackbar("MSER_area_treshhold", "MSER TEST",   &MSER_area_treshhold, 500, on_trackbar);
    createTrackbar("MSER_min_marign", "MSER TEST",   &MSER_min_marign, 500, on_trackbar);
    createTrackbar("MSER_edge_blur_stage", "MSER TEST",   &MSER_edge_blur_stage, 100, on_trackbar);
    while(true){
    //createButton("Next image", on_button, 0, CV_PUSH_BUTTON);
        on_trackbar(MSER_min_area, 0);
        waitKey(0);
        on_button(num, on_trackbar);
    }
}

int ksize=0, d=9, sigmacolor=20, sigmaspace=30, clipLimit=5, need_hist=0;

int sigma1=5, sigma2=36, sigma3=86;
int power=10;

cv::Mat multiScaleRetinex(const cv::Mat& image, const std::vector<double>& sigmas) {
    cv::Mat result = cv::Mat::zeros(image.size(), CV_64F);
    cv::Mat logImage;
    image.convertTo(logImage, CV_64F);

    vector<std::future<cv::Mat> >tem_res;

    // Применение MSR с различными масштабами
    for (double sigma : sigmas) {
        // Сглаживание изображения с Гауссовым ядром
        cv::Mat smoothed;

        if (!sigma) smoothed=Mat::zeros(logImage.size(), CV_64F);
        else
            cv::GaussianBlur(logImage, smoothed, cv::Size(0, 0), sigma);

        // Вычитание сглаженного изображения из логарифма исходного
        cv::Mat diff = logImage - smoothed;

        // Нормализация результатов
        cv::normalize(diff, diff, 0, 255, cv::NORM_MINMAX);

        // Добавление к результату
        result += diff;
    }
    result/=sigmas.size();
    cv::pow(result, power/10.0, result);

    // Нормализация окончательного изображения и преобразование обратно в 8-битный формат
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    result.convertTo(result, CV_8U);

    return result;
}



MyFaceRecognition FR;








Point2f MiddlePoint(vector<Point2f> points){
    Point2f answer(0,0);
    for (int i=0; i<points.size(); ++i){
        answer+=(points[i]-answer)/double(i+1);
    }
    return answer;
}
/*
double FromPerspective(double k, double l, double r, double x, double precision=0.001){
    double ans=0;
    double m;
    while(true){
        if (x<l){
            k*=k;
            double w=(r-l);
            r=l;
            ans-=1.0;
            l-=w*k;

        }else if (x>r){
            k*=k;
            double w=(r-l);
            l=r;
            ans+=1.0;
            r+=w/k;
        }else break;
    }
    //cout << "(l/x/r): " << l << "/" << x << "/" << r << endl;
    for(int i=1; ; ++i){
        //cout << "ans: " << std::setprecision(4) << ans << endl;
        m=r+(l-r)/(k+1);
        if (x>=m){
            l=m;
            double add=pow<double>(2,-i);
            if (add<precision) return ans;
            ans+=add;
        }else if(m>x){
            r=m;
        }
        k=sqrt(k);
    }
}

double ToPerspective(double k, double l, double r, double x, double precision=0.001){

    for (double step=0.5; step>precision; step/=2.0){
        if (x>=step){
            l+=(r-l)/(k+1)*k;
            x-=step;
            if (x<precision) return l;
        }else if (x<step){
            r=r+(l-r)/(k+1);
        }
    }
    return l;
}

Point2f unperspectivePoint(double k, Point2f dl, Point2f dr, Point2f ul, Point2f ur, Point2f P){
    Point2f pH=PointAcross2Lines(dl, dr, ul, ur);
    double localx=FromPerspective(k, dl.x, dr.x, P.x);
    Point2f pl=PointAcross2Lines(P, pH, Point2f(dl.x, 0), Point2f(dl.x, 10));
    double localy=(pl.y-ul.y)/(dl.y-ul.y);
    Point2f answer(localx, localy);
    return answer;
}

short sign(double num){
    if (num<0) return -1;
    if (num>0) return 1;
    return 0;
}
*/


static void on_trackbar_noise(int, void*){
VideoCapture capture(1);
Mat output;
while (true){
    capture >> output;
     //NORAMLS[num].copyTo(output);

    //imshow("input", NORAMLS[i]);
    Mat tempo;
    output.copyTo(tempo);
    cv::bilateralFilter(output, tempo, d, sigmacolor, sigmaspace);
    tempo.copyTo(output);

    //imshow("Denoise + retinex", output);

    FR.retinexMSR(output, {sigma1, sigma2});

    auto rects=FR.GetFaceRects(output);
    if (rects.size()==0) return;//continue;
    /*

    Mat tempM; output.copyTo(tempM);
    cv::cvtColor(tempM, tempM, cv::COLOR_RGB2GRAY);
            cv::cvtColor(tempM, tempM, cv::COLOR_GRAY2RGB);
    for (int f=0; f<rects.size(); ++f)
       rectangle(tempM, rects[f], Scalar(0, 255,255), 3);
    imshow("previos Rects", tempM);
    */
    vector<pair<Mat, Rect>> Faces=FR.NormalizeRotate(output, rects);
    Mat Face;
    if (Faces.size()==0){ Face=Mat::zeros(512,512, CV_8UC3); }
    else Face=Faces[0].first;
    //return;
    //imshow("Face previos", Face);
    rects=FR.GetFaceRects(Face);
    /*
    Face.copyTo(tempM);
    cv::cvtColor(tempM, tempM, cv::COLOR_RGB2GRAY);
            cv::cvtColor(tempM, tempM, cv::COLOR_GRAY2RGB);
    for (int f=0; f<rects.size(); ++f)
       rectangle(tempM, rects[f], Scalar(0, 255,255), 3);

    imshow("last Rects", tempM);
    */
    if (rects.size()==0) rects={Rect(0,0, Face.cols, Face.rows)};
    Faces=FR.NormalizeRotate(Face, rects);

    if (Faces.size()!=0) Face=Faces[0].first;

    //imshow("Face last", Face);


    Face=FR.HorizontalPerpectiveNormalize(Face);
    cout << FR.calculateMeanBrightness(Face) << endl;
    double brightnessOffset = 110 - FR.calculateMeanBrightness(Face);
    Face.convertTo(Face, -1, 1, brightnessOffset);
    imshow("Unperspective", Face);
    waitKey();
}
/*
    //imshow("Denoise TEST", output);
    //return;

    //imshow("MSER TEST", temp);



    //resize(output, output, Size(512,512), 0,0, INTER_CUBIC);

    std::vector<double> sigmas = {10};
    //Mat Restinx1 = multiScaleRetinex(output, sigmas);
    //Mat Restinx1 = SSR_retinex(output, 0.5);
    //sigmas = {5, 512};
    //Mat Restinx2 = multiScaleRetinex(output, sigmas);
    //Mat Restinx2 = SSR_retinex(output, 5);

    //Mat Restinx3 = multiScaleRetinex(output, sigmas);
    //Mat Restinx3 = SSR_retinex(output, 10);

    //cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);
    //try{
        //tempo;
        output.copyTo(tempo);
        cv::bilateralFilter(output, tempo, d, sigmacolor, sigmaspace);
        //cv::bilateralFilter(Restinx2, Restinx2, d, sigmacolor, sigmaspace);
        //cv::bilateralFilter(Restinx3, Restinx3, d, sigmacolor, sigmaspace);

        sigmas = {double(sigma2)};
        Mat Restinx1 = multiScaleRetinex(tempo, sigmas);
        double brightnessOffset = 110 - FR.calculateMeanBrightness(Restinx1);
        Restinx1.convertTo(Restinx1, -1, 1, brightnessOffset);
        if (need_hist) equalizeHist(Restinx1, Restinx1);
        sigmas = {double(sigma1), double(sigma2)};
        Mat Restinx2 = multiScaleRetinex(tempo, sigmas);
        brightnessOffset = 110 - FR.calculateMeanBrightness(Restinx2);
        Restinx2.convertTo(Restinx2, -1, 1, brightnessOffset);

        if (need_hist) equalizeHist(Restinx2, Restinx2);
        sigmas = {double(sigma1), double(sigma2), double(sigma3)};
        Mat Restinx3 = multiScaleRetinex(tempo, sigmas);

        if (need_hist) equalizeHist(Restinx3, Restinx3);
    //cv::medianBlur(output, output, ksize);
        imshow("Denoise TEST gray", output);
        resize(Face, output, Size(512,512), 0,0, INTER_CUBIC);
        cout << FR.calculateMeanBrightness(Restinx2) << endl;

        Mat R1=((Restinx1-output)*2+122);
        //cv::bilateralFilter(Restinx3, tempo, d, sigmacolor, sigmaspace);
        //tempo.copyTo(Restinx3);

        Mat R2=((Restinx2-output)*2+122);

        Mat R3=((Restinx3-output)*2+122);
        imshow("Denoise TEST", output);

        hconcat(Restinx1, R1, Restinx1);
        imshow("Denoise R1", Restinx1);
        hconcat(Restinx2, R2, Restinx2);
        imshow("Denoise R2", Restinx2);
        hconcat(Restinx3, R3, R3);
        imshow("Denoise R3", R3);
    //}catch(...){
        //if (ksize==0)
        //    imshow("Denoise TEST", output);
        //else
        //imshow("Denoise TEST", Mat::zeros(output.rows, output.cols, CV_8UC1));
    //}
*/
}





void TestDeNoise(){

    string temp;
    ifstream file("/home/nikita/QtProj/GetPictureStudents/Faces/people.csv");
    for (int i=0; i<2151; ++i){
        getline(file, temp);
        NORAMLS[i]=imread(temp);
        Mat output;
        NORAMLS[i].copyTo(output);
        imshow("input", NORAMLS[i]);

        Mat tempo;
        //output.copyTo(tempo);
        cv::bilateralFilter(output, tempo, d, sigmacolor, sigmaspace);
        tempo.copyTo(output);
        output=FR.retinexMSR(output, {sigma1, sigma2});
        imshow("Denoise + retinex", output);
        //waitKey();
        auto rects=FR.GetFaceRects(output);
        if (rects.size()==0) continue;//return;//


        Mat tempM; output.copyTo(tempM);
        cv::cvtColor(tempM, tempM, cv::COLOR_RGB2GRAY);
                cv::cvtColor(tempM, tempM, cv::COLOR_GRAY2RGB);
        for (int f=0; f<rects.size(); ++f)
           rectangle(tempM, rects[f], Scalar(0, 255,255), 3);
        imshow("previos Rects", tempM);

        vector<pair<Mat, Rect>> Faces=FR.NormalizeRotate(output, rects);
        Mat Face; Rect faceRect;
        if (Faces.size()==0){ Face=Mat::zeros(512,512, CV_8UC3); }
        else{ Face=Faces[0].first; faceRect=Faces[0].second;}
        imshow("input2", Face);
        //waitKey();


        Face=FR.HorizontalPerpectiveNormalize(Face, faceRect);
        cout << FR.calculateMeanBrightness(Face) << endl;
        double brightnessOffset = 113 - FR.calculateMeanBrightness(Face);
        Face.convertTo(Face, -1, 1, brightnessOffset);
        imshow("Unperspective", Face);
        cvtColor(Face, Face, cv::COLOR_BGR2GRAY);
        imshow("gray", Face);
        waitKey();

    }
    namedWindow("Denoise TEST");

    //createTrackbar("ksize", "Denoise TEST",           &ksize, 100, on_trackbar_noise);


    createTrackbar("d", "Denoise TEST",           &d, 100, on_trackbar_noise);
    createTrackbar("sigmacolor", "Denoise TEST",           &sigmacolor, 512, on_trackbar_noise);
    createTrackbar("sigmaspace", "Denoise TEST",           &sigmaspace, 512, on_trackbar_noise);

    createTrackbar("clipLimit", "Denoise TEST",           &clipLimit, 100, on_trackbar_noise);

    createTrackbar("ksize", "Denoise TEST",           &ksize, 100, on_trackbar_noise);
    createTrackbar("need equalize hist", "Denoise TEST",           &need_hist, 1, on_trackbar_noise);

    createTrackbar("ksize1_MSR", "Denoise TEST",           &sigma1, 500, on_trackbar_noise);
    createTrackbar("ksize2_MSR", "Denoise TEST",           &sigma2, 500, on_trackbar_noise);
    createTrackbar("ksize3_MSR", "Denoise TEST",           &sigma3, 512, on_trackbar_noise);
    createTrackbar("power_MSR", "Denoise TEST",           &power, 30, on_trackbar_noise);

    while(true){
    //createButton("Next image", on_button, 0, CV_PUSH_BUTTON);
        on_trackbar_noise(d, 0);
        waitKey(0);
        on_button(num, on_trackbar_noise);

    }
}

void Test(){

    VideoCapture capture(0);
    Mat output;
    Mat temp;
    while (true){
        capture >> temp ;
        temp.copyTo(output);
        if (temp.empty()) continue;
         //NORAMLS[num].copyTo(output);

        //imshow("input", NORAMLS[i]);
        Mat tempo;
        //output.copyTo(tempo);
        cv::bilateralFilter(output, tempo, d, sigmacolor, sigmaspace);
        tempo.copyTo(output);


        output=FR.retinexMSR(output, {sigma1, sigma2});
        imshow("Denoise + retinex", output);

        vector<Rect> rects=FR.GetFaceRects(output);
        if (rects.size()!=0){//return;//
            /*

            Mat tempM; output.copyTo(tempM);
            cv::cvtColor(tempM, tempM, cv::COLOR_RGB2GRAY);
                    cv::cvtColor(tempM, tempM, cv::COLOR_GRAY2RGB);
            for (int f=0; f<rects.size(); ++f)
               rectangle(tempM, rects[f], Scalar(0, 255,255), 3);
            imshow("previos Rects", tempM);
            */
            vector<pair<Mat, Rect>> Faces=FR.NormalizeRotate(output, rects);
            Mat Face; Rect faceRect;
            if (Faces.size()==0){ Face=Mat::zeros(512,512, CV_8UC3); }
            else{
                Face=Faces[0].first; faceRect=Faces[0].second;
                Face=FR.HorizontalPerpectiveNormalize(Face, faceRect);
            }


            //imshow("Perspective scene", Face);
        }
        if (waitKey(40)==27) break;
        //return;
    }
}


int main(int argc, const char *argv[]) {

    FR.PrepareDataset("/home/nikita/QtProj/GetPictureStudents/Faces/people.csv");
    //Test();
    //TestDeNoise();
    TestMSER();
    //CelebrityTest();
    //return 0;
    FR.PrepareDataset("/home/nikita/QtProj/GetPictureStudents/Faces/people.csv");
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
