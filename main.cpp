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

int ksize=8, d=9, sigmacolor=19, sigmaspace=7, clipLimit=5, need_hist=0;

cv::Mat multiScaleRetinex(const cv::Mat& image, const std::vector<double>& sigmas) {
    cv::Mat result = cv::Mat::zeros(image.size(), CV_64F);
    cv::Mat logImage, logmat;
    image.convertTo(logImage, CV_64F);
    //cv::log(logImage, logmat);
    // Применение MSR с различными масштабами
    for (double sigma : sigmas) {
        // Сглаживание изображения с Гауссовым ядром
        cv::Mat smoothed;
        cv::GaussianBlur(logImage, smoothed, cv::Size(0, 0), sigma);
        //cv::Mat logGaus; cv::log(smoothed, logGaus);
        //logImage-=logGaus;
        //logImage/=log(10);
        // Вычитание сглаженного изображения из логарифма исходного
        cv::Mat diff = logImage - smoothed;

        // Нормализация результатов
        cv::normalize(diff, diff, 0, 255, cv::NORM_MINMAX);

        // Добавление к результату
        result += logImage;

    }

    // Нормализация окончательного изображения и преобразование обратно в 8-битный формат
    //result/=sigmas.size();

    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    //cv::exp(result, result);
    result.convertTo(result, CV_8U);

    return result;
}
cv::Mat multiScaleRetinex_color(const cv::Mat& image, const std::vector<double>& sigmas) {
    Mat chanel_src[3], channel_dst[3];
    split(image, chanel_src);
    std::future<Mat> chanels[3];
    for (int i=0; i<3; ++i)
        chanels[i]=async(std::launch::async, multiScaleRetinex, chanel_src[i], sigmas);
    for (int i=0; i<3; ++i)
        channel_dst[i]=chanels[i].get();
    Mat result;
    merge(channel_dst, 3, result);
    return result;
}


cv::Mat SSR_retinex(const cv::Mat &image, double sigma) {
    // Шаг 1: Разложение изображения на логарифмы
    cv::Mat imageLog;
    image.convertTo(imageLog, CV_64F);
    cv::log(imageLog, imageLog);

    // Шаг 2: Фильтрация изображения для получения отраженного освещения
    cv::Mat filtered;
    cv::GaussianBlur(imageLog, filtered, cv::Size(0, 0), sigma);

    // Вычитание отраженного освещения из исходного логарифма
    cv::Mat reflection = imageLog - filtered;

    // Шаг 3: Нормализация отраженного освещения
    cv::normalize(reflection, reflection, 0, 255, cv::NORM_MINMAX);

    // Шаг 4: Обратное преобразование логарифма отраженного освещения
    cv::exp(reflection, reflection);

    // Шаг 5: Суммирование отраженного освещения с логарифмом источника света
    cv::Mat illumination = imageLog - reflection;

    // Шаг 6: Обратное преобразование для получения окончательного изображения
    cv::Mat result;
    cv::exp(illumination, result);

    // Нормализация окончательного изображения и конвертация обратно в 8-битный формат
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    result.convertTo(result, CV_8U);

    return result;
}

cv::Mat retinexMSR(const cv::Mat &src, const std::vector<double> &weights, const std::vector<int> &scales) {
    CV_Assert(src.channels() == 3);  // Проверка на цветное изображение
    Mat tempo_src; src.copyTo(tempo_src);
    cv::Mat srcLog;
    tempo_src.convertTo(srcLog, CV_32FC3);
    //cv::log(srcLog, srcLog);
    //imshow("log src", srcLog);
    cv::Mat sumMat = cv::Mat::zeros(src.size(), CV_32FC3);
    //src.copyTo(srcLog);
    //cv::cvtColor(tempo_src, tempo_src, cv::COLOR_BGR2Lab);
    //Mat src_ch[3]; split(srcLog)

    std::vector<cv::Mat> logscales;
    //tempo_src.convertTo(srcLog, CV_32FC3);
    //cv::log(srcLog, srcLog);

    //srcLog/=log(10);
    for (size_t i = 0; i < scales.size(); ++i) {
        cv::Mat imgBlur;
        cv::GaussianBlur(srcLog, imgBlur, cv::Size(0,0), scales[i]);

        //imgBlur.convertTo(imgBlur, CV_32FC3);
        //cv::log(imgBlur, imgBlur);
        //imgBlur/=log(10);
        logscales.push_back(srcLog - imgBlur);
        //cv::exp(logscales[i], logscales[i]);
        sumMat += logscales[i] * weights[i];
    }
    //cv::exp(sumMat, sumMat);
    sumMat.convertTo(sumMat, CV_8UC3);

    //cv::Mat dst;
    //cv::cvtColor(sumMat, sumMat, cv::COLOR_Lab2BGR);

    //cv::normalize(sumMat, sumMat, 0, 255, cv::NORM_MINMAX);

    return sumMat;
}

static void on_trackbar_noise(int, void*){

    Mat output; NORAMLS[num].copyTo(output);

    //imshow("MSER TEST", temp);



    resize(output, output, Size(512,512), 0,0, INTER_CUBIC);

/*
    std::vector<double> weights = {1}; // Равные веса для простоты
        std::vector<double> sigmas_color = {5}; // Разные масштабы для SSR
    cv::Mat result = multiScaleRetinex_color(output, sigmas_color);
    result = multiScaleRetinex_color(result, sigmas_color);
    //result=(result-output)+output;
            //(result-output);//*3+output;
    hconcat(output, result-output+122, output);
    hconcat(output, result, result);
    imshow("Denoise TEST", result);
    Mat chanels[3]; split(result, chanels);
    //imshow("Denoise 0", chanels[0]);
    //imshow("Denoise 1", chanels[1]);
    //imshow("Denoise 2", chanels[2]);

    return;
*/

    cv::Mat grays[3];
    cv::cvtColor(output, output, cv::COLOR_RGB2HSV);
    split(output, grays);
    output=grays[2];

    std::vector<double> sigmas = {5};
    Mat Restinx1 = multiScaleRetinex(output, sigmas);
    //Mat Restinx1 = SSR_retinex(output, 0.5);
    sigmas = {15, 80};
    Mat Restinx2 = multiScaleRetinex(output, sigmas);
    //Mat Restinx2 = SSR_retinex(output, 5);
    sigmas = {15, 80, 250, 512};
    Mat Restinx3 = multiScaleRetinex(output, sigmas);
    //Mat Restinx3 = SSR_retinex(output, 10);

    //cv::normalize(output, output, 0, 255, cv::NORM_MINMAX);
    //try{
        Mat tempo;
        output.copyTo(tempo);
        //if (need_hist) equalizeHist(tempo, tempo);
        //cv::bilateralFilter(output, tempo, d, sigmacolor, sigmaspace);
        //cv::bilateralFilter(Restinx2, Restinx2, d, sigmacolor, sigmaspace);
        //cv::bilateralFilter(Restinx3, Restinx3, d, sigmacolor, sigmaspace);
        if (ksize!=0){
            auto CLAHE = createCLAHE(clipLimit, Size(ksize, ksize));

            //CLAHE->apply(tempo, tempo);
            //CLAHE->apply(Restinx, Restinx);

        }
    //cv::medianBlur(output, output, ksize);
        imshow("Denoise TEST gray", output);
        Mat R1=((output-Restinx1)*2+122);
        Mat R2=((output-Restinx2)*2+122);
        Mat R3=((output-Restinx3)*2+122);
        resize(NORAMLS[num], output, Size(512,512), 0,0, INTER_CUBIC);
        imshow("Denoise TEST", output);

        hconcat(Restinx1, R1, Restinx1);
        imshow("Denoise R1", Restinx1);
        hconcat(Restinx2, R2, Restinx2);
        imshow("Denoise R2", Restinx2);
        hconcat(Restinx3, R3, Restinx3);
        imshow("Denoise R3", Restinx3);
    //}catch(...){
        //if (ksize==0)
        //    imshow("Denoise TEST", output);
        //else
        //imshow("Denoise TEST", Mat::zeros(output.rows, output.cols, CV_8UC1));
    //}

}

void TestDeNoise(){
    string temp;
    ifstream file("/home/nikita/QtProj/GetPictureStudents/Faces/people.csv");
    for (int i=0; i<2151; ++i){
        getline(file, temp);
        NORAMLS[i]=imread(temp);
    }
    namedWindow("Denoise TEST");

    //createTrackbar("ksize", "Denoise TEST",           &ksize, 100, on_trackbar_noise);


    //createTrackbar("d", "Denoise TEST",           &d, 100, on_trackbar_noise);
    //createTrackbar("sigmacolor", "Denoise TEST",           &sigmacolor, 512, on_trackbar_noise);
    //createTrackbar("sigmaspace", "Denoise TEST",           &sigmaspace, 512, on_trackbar_noise);
    //createTrackbar("clipLimit", "Denoise TEST",           &clipLimit, 100, on_trackbar_noise);

    //createTrackbar("ksize", "Denoise TEST",           &ksize, 100, on_trackbar_noise);
    //createTrackbar("need equalize hist", "Denoise TEST",           &need_hist, 1, on_trackbar_noise);
    while(true){
    //createButton("Next image", on_button, 0, CV_PUSH_BUTTON);
        on_trackbar_noise(d, 0);
        waitKey(0);
        on_button(num, on_trackbar_noise);

    }
}



int main(int argc, const char *argv[]) {
    TestDeNoise();
    //TestMSER();
    //CelebrityTest();
    //return 0;
    MyFaceRecognition FR;
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
