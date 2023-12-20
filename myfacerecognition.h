#ifndef MYFACERECOGNITION_H
#define MYFACERECOGNITION_H

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;
using namespace cv::face;

class MyFaceRecognition
{
public:
    MyFaceRecognition();
    bool Init(string, string="");
    void NormalizeImage(Mat& input, Mat& output, vector<Point2f> &shapes, Rect faceRec=Rect(0,0,0,0), uint c=0);
    void PrepareDataset(string FileData);
    bool ExtractFace(Mat input, Mat& output, double total=0.0, uint c=0);
    vector<pair<int, double> > GetSimilarFacesLBPH(Mat& ProcessingImage, int count, double tresh=30.0, uint n=0);
    double ElasticGraphMath(vector<cv::Point> elasticGraph1, int label, Mat& Mathed, double coff=1.0);
    static pair<bool, double> compareFacesWithSIFT_RANSAC(Mat, Mat, vector<KeyPoint>, vector<KeyPoint>, double=0.0);

    vector<DMatch> _compareFacesWithSIFT(Mat &descriptorsReference, Mat &descriptorsTest, double=0.0){
        if (descriptorsReference.empty() || descriptorsTest.empty() || descriptorsReference.size() != descriptorsTest.size() || descriptorsReference.type() != descriptorsTest.type()) {
            cout << descriptorsReference.empty() << " " << descriptorsTest.empty() << " " << (descriptorsReference.size() != descriptorsTest.size()) << " " << (descriptorsReference.type() != descriptorsTest.type()) << "\n";
            cout.flush();
            return {};
        }
/*
        BFMatcher matcher;
        vector<DMatch> matches;
        matcher.match(descriptorsReference, descriptorsTest, matches);
        sort(matches.begin(), matches.end(), [](cv::DMatch p1, cv::DMatch p2){return p1.distance > p2.distance;});
        matches.resize(200);
        vector<DMatch> filter_matches;

        double totalDistance = 0.0;
        for (const cv::DMatch& match : matches) {
            if (match.distance<0.043
                ){
                filter_matches.push_back(match);
                totalDistance += match.distance;
            }
        }
        return filter_matches;
*/

        cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descriptorsReference, descriptorsTest, knnMatches, 200);
        const float ratioThreshold = 0.9f;
        std::vector<cv::DMatch> goodMatches;
        for (const auto& match : knnMatches) {
            if (match[0].distance < ratioThreshold * match[1].distance) {
                goodMatches.push_back(match[0]);
            }
        }
        return goodMatches;

        //vector<pair
    }

    inline Mat getMatByLabel(int label){
        return *(images.begin()+distance(labels.begin(), find(labels.begin(), labels.end(), label)));
    }

    inline Mat getDescriptByLabel(int label){
        return *(descriptors.begin()+distance(labels.begin(), find(labels.begin(), labels.end(), label)));
    }

    inline vector<KeyPoint> getKeyPointByLabel(int label){
        return *(keypoints.begin()+distance(labels.begin(), find(labels.begin(), labels.end(), label)));
    }

    static void some(Mat& image, Mat& output);

    Ptr<LBPHFaceRecognizer> model;


private:
    static vector<KeyPoint> getBasePoint(vector<cv::KeyPoint> kps) {
        vector<pair<double, uint>> diffs(kps.size());
        sort(kps.begin(), kps.end(), [](const KeyPoint& kp1, const KeyPoint& kp2){return kp1.angle<kp2.angle;});
        for (uint i=1; i<kps.size(); i++){
            diffs[i]={abs(kps[i].angle-kps[i-1].angle),i};
        }
        diffs[0]={abs(kps[0].angle-kps[kps.size()-1].angle),0};
        sort(diffs.begin(), diffs.end());
        vector<KeyPoint> answer;
        for (auto now: diffs){
            answer.push_back(kps[now.second]);
        }
        return answer;
    }


    const double theta=0.0, sigma=1, frequency=0.5;
    double gaborFilter(double x, double y){
        double xPrime = x * cos(theta) + y * sin(theta);
        double yPrime = -x * sin(theta) + y * cos(theta);
        double exponent = -(xPrime * xPrime + yPrime * yPrime) / (2 * sigma * sigma);
        double cosValue = cos(2 * M_PI * frequency * xPrime);
        return exp(exponent) * cosValue;
    }

    double compareElasticGraphs(const std::vector<cv::Point>& graph1,
                                const std::vector<cv::Point>& graph2) {
        // Проверка на равное количество точек в графах
        if (graph1.size() != graph2.size()) {
                std::cerr << "Ошибка: графы имеют разное количество точек" << std::endl;
                return 0.0;
            }

            int numPoints = graph1.size();
            std::vector<std::vector<double>> dp(numPoints, std::vector<double>(numPoints, 0.0));

            // Вычисление значений сходства для всех пар точек с использованием динамического программирования
            for (int i = 0; i < numPoints; ++i) {
                for (int j = 0; j < numPoints; ++j) {
                    double x1 = graph1[i].x;
                    double y1 = graph1[i].y;

                    double x2 = graph2[j].x;
                    double y2 = graph2[j].y;

                    double gaborValue = gaborFilter(x2 - x1, y2 - y1);

                    // Вычисление значения сходства с учетом предыдущих значений
                    if (i > 0 && j > 0) {
                        dp[i][j] = gaborValue + std::max(dp[i-1][j-1], std::max(dp[i-1][j], dp[i][j-1]));
                    } else {
                        dp[i][j] = gaborValue;
                    }
                }
            }

            // Возвращение значения сходства для последних точек в графах
            return dp[numPoints-1][numPoints-1];
        }


    static bool KeyPointCompare(KeyPoint p1, KeyPoint p2){
        return p1.response > p2.response;
    }

    std::vector<cv::Point> createElasticGraphORB(const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& image) {
        std::vector<cv::Point> elasticGraph;

        // Параметры для построения эластического графа
        int numPoints = 70;  // Количество точек в графе
        double elasticFactor = 0.4;  // Фактор эластичности

        // Отсортировать ключевые точки по их релевантности (например, по score или response)
        std::vector<cv::KeyPoint> sortedKeypoints = keypoints;
        sort(sortedKeypoints.begin(), sortedKeypoints.end(), KeyPointCompare);
        // Отсортировать ключевые точки по убыванию их релевантности

        // Выбрать первые numPoints ключевых точек
        sortedKeypoints.resize(numPoints);

        // Построить эластический граф на основе выбранных ключевых точек
        for (const cv::KeyPoint& keypoint : sortedKeypoints) {
            cv::Point point = keypoint.pt;

            // Применить эластичность к координатам точки
            point.x += elasticFactor * keypoint.response;
            point.y += elasticFactor * keypoint.response;

            // Проверить, что точка находится внутри изображения
            if (point.x >= 0 && point.x < image.cols && point.y >= 0 && point.y < image.rows) {
                elasticGraph.push_back(point);
            }
        }

        return elasticGraph;
    }

    std::vector<cv::Point> createElasticGraphSIFT(const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors, const cv::Mat& image) {
        std::vector<cv::Point> elasticGraph;

        // Параметры для построения эластического графа
        int numPoints = 70;  // Количество точек в графе
        double elasticFactor = 0.4;  // Фактор эластичности

        // Отсортировать ключевые точки по их релевантности (например, по score или response)
        std::vector<cv::KeyPoint> sortedKeypoints = keypoints;
        sort(sortedKeypoints.begin(), sortedKeypoints.end(), KeyPointCompare);
        // Отсортировать ключевые точки по убыванию их релевантности

        // Выбрать первые numPoints ключевых точек
        sortedKeypoints.resize(numPoints);

        // Построить эластический граф на основе выбранных ключевых точек и их дескрипторов
        for (size_t i = 0; i < sortedKeypoints.size(); ++i) {
            cv::Point point = sortedKeypoints[i].pt;

            // Применить эластичность к координатам точки, используя информацию из дескриптора
            double elasticOffset = elasticFactor * sortedKeypoints[i].response;
            point.x += elasticOffset * descriptors.at<float>(i, 0); // Используем первую компоненту дескриптора
            point.y += elasticOffset * descriptors.at<float>(i, 1); // Используем вторую компоненту дескриптора

            // Проверить, что точка находится внутри изображения
            if (point.x >= 0 && point.x < image.cols && point.y >= 0 && point.y < image.rows) {
                elasticGraph.push_back(point);
            }
        }

        return elasticGraph;
    }

    cv::dnn::Net net = cv::dnn::readNetFromCaffe("/home/nikita/Загрузки/deploy.prototxt", "/home/nikita/Загрузки/res10_300x300_ssd_iter_140000_fp16.caffemodel");
    cv::Ptr<cv::face::Facemark> facemark = cv::face::FacemarkLBF::create();
    vector<Mat> images;
    vector<int> labels;
    vector<Mat> descriptors;
    vector<vector<KeyPoint>> keypoints;

    void read_csv(const string& filename,
                         vector<Mat> &images,
                         vector<int> &labels, bool Grayscale=true) {

        std::ifstream file(filename.c_str(), ifstream::in);
        if (!file) {
            string error_message = "No valid input file was given, please check the given filename.";
            CV_Error(Error::StsBadArg, error_message);
        }
        string line, path, classlabel;
        while (!file.eof()) {
            try{
            file >> path;
            if (path.empty()) break;
            file >> classlabel;
            cout << "|" << path << "| |" << classlabel << "|\n";
            }catch(Exception e){
            cout << e.what() << " " << e.code << " " << e.err << " " << e.line << e.msg << "\n";
            }

            if(!path.empty() && !classlabel.empty()) {
                try{
                    if (Grayscale)
                        images.push_back(imread(path, IMREAD_GRAYSCALE));
                    else
                        images.push_back(imread(path));
                    //
                    labels.push_back(atoi(classlabel.c_str()));
                }catch(...){
                }
            }

        }
    }
};

#endif // MYFACERECOGNITION_H
