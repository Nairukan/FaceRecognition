#include "myfacerecognition.h"

#include <future>

#include <opencv2/xfeatures2d/nonfree.hpp>

MyFaceRecognition::MyFaceRecognition(){


}

void MyFaceRecognition::some(Mat& image, Mat& output){
    // Преобразование изображения в оттенки серого
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Улучшение контраста изображения

    // Нормализация яркости изображения
    //cv::normalize(grayImage, grayImage, 0, 255, cv::NORM_MINMAX);

    // Увеличение резкости изображения
    cv::Mat sharpenedImage;
    cv::GaussianBlur(grayImage, sharpenedImage, cv::Size(0, 0), 3);
    cv::addWeighted(grayImage, 1.5, sharpenedImage, -0.5, 0, sharpenedImage);

    // Применение гауссова размытия для уменьшения шума
    cv::GaussianBlur(sharpenedImage, sharpenedImage, cv::Size(3, 3), 0);
    normalize(sharpenedImage, sharpenedImage, 0, 255, NORM_MINMAX);
    // Применение адаптивной бинаризации для выделения контуров
    //cv::Mat binaryImage;
    //cv::adaptiveThreshold(sharpenedImage, binaryImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);
    //cv::equalizeHist(grayImage, grayImage);

    // Показ изображений (для отладки)
    output=sharpenedImage;
}

void MyFaceRecognition::PrepareDataset(string FileData){
    //net=cv::dnn::readNetFromCaffe("/home/nikita/Загрузки/deploy.prototxt", "/home/nikita/Загрузки/res10_300x300_ssd_iter_140000_fp16.caffemodel");
    //facemark->loadModel("/home/nikita/GSOC2017/data/lbfmodel.yaml");
    facemark->loadModel("/home/nikita/QtProj/build-TrainLBF-Desktop-Release/MyModel3.yaml");
    facemark->setFaceDetector(defaultFaceDetector, (void*)nullptr);
return;
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    func_get_image(FileData);
    string fn_csv = "LBPH_people.csv";
    try {
        read_csv(fn_csv, images, labels, false);
    } catch (const cv::Exception& e) {
        cerr << "Ошибка открытия файла \"" << fn_csv << "\". Причина: " << e.msg << endl;
        exit(1);
    }
    if(images.size() <= 1) {
        string error_message = "Недостаточно изображений в базе!";
        CV_Error(Error::StsError, error_message);
    }
    int MSER_delta=5, MSER_min_area=60, MSER_max_area=14400; double MSER_max_variation=0.25, MSER_min_diversity=0.200000000001;
    int MSER_max_evolution=200; double MSER_area_treshhold=1.01, MSER_min_marign=0.0030000001; int MSER_edge_blur_stage=5;

    MSER_delta=1; MSER_min_area=200; MSER_max_area=3000;
    MSER_max_variation=0.8; MSER_min_diversity=0.1;
    MSER_area_treshhold=2; MSER_edge_blur_stage=18;
    MSER_max_evolution=1000; MSER_min_marign=0.01;



    Ptr<MSER> mser = MSER::create(MSER_delta, MSER_min_area, MSER_max_area, MSER_max_variation, MSER_min_diversity, MSER_max_evolution, MSER_area_treshhold, MSER_min_marign, MSER_edge_blur_stage);

    //Ptr<BRISK> brisk = BRISK::create(BRISK_tresh, BRISK_octaves, BRISK_patternScale);
    Ptr<xfeatures2d::SURF> sift = xfeatures2d::SURF::create(300);

    //Ptr<AKAZE> akaze = AKAZE::create();
    //sift->setNFeatures(100);

    ofstream ouf ("Norm_Face/Faces.csv");
    //future<string> RE[images.size()];
    string RE[images.size()];
    int d=9, sigmacolor=20, sigmaspace=30, clipLimit=5, need_hist=0;

    int sigma1=5, sigma2=36, sigma3=86;
    int power=10;
    for (int i=0; i<images.size(); i++){

        Mat tempo, output;
        images[i].copyTo(output);
        //output.copyTo(tempo);
        cv::bilateralFilter(output, tempo, d, sigmacolor, sigmaspace);
        tempo.copyTo(output);
        output=retinexMSR(output, {sigma1, sigma2});
        //imshow("Denoise + retinex", output);
        //waitKey();
        auto rects=GetFaceRects(output);
        if (rects.size()==0) continue;//return;//



        //imshow("previos Rects", tempM);

        vector<pair<Mat, Rect>> Faces=NormalizeRotate(output, rects);
        Mat Face; Rect faceRect;
        if (Faces.size()==0){ Face=Mat::zeros(512,512, CV_8UC3); }
        else{ Face=Faces[0].first; faceRect=Faces[0].second;}
        //imshow("input2", Face);
        //waitKey();


        Face=HorizontalPerpectiveNormalize(Face, faceRect);
        //cout << calculateMeanBrightness(Face) << endl;
        double brightnessOffset = 113 - calculateMeanBrightness(Face);
        Face.convertTo(Face, -1, 1, brightnessOffset);
        //imshow("Unperspective", Face);
        cvtColor(Face, Face, cv::COLOR_BGR2GRAY);
        Face.copyTo(images[i]);
        //waitKey();

        cout << i << "\n"; cout.flush();/*
        //RE[i] = std::async(std::launch::async, [=]{
            std::vector<cv::KeyPoint> keypoints;
            std::vector<Mat> descriptors;

            //resize(images[i], images[i], Size(512, 512));
            //some(images[i], images[i]);
            //for (uint q=0; q<=0; ++q){

                Mat tempo;
                images[i].copyTo(tempo);
                //mser->setMinDiversity(MSER_min_diversity+0.5*q);
                //mser->setMinMargin(MSER_min_marign+0.5*q);
                //mser.reset();
                vector<vector<Point>> regions;
                vector<Rect> mserBoundBoxes;
                mser->detectRegions(tempo, regions, mserBoundBoxes);
                long double sum=0.0;
                for (const auto& region : regions){
                    sum+=region.size();
                }
                long double coef=350/sum;
                cv::Mat resultImage2;
                cv::cvtColor(tempo, resultImage2, cv::COLOR_GRAY2BGR);
                for (const auto& region : regions){
                    vector<Point> contour(region); // Convert MSER region to contour
                            vector<vector<Point>> contours;
                            contours.push_back(contour);
                            //drawContours(resultImage2, contours, 0, Scalar(0), 2); // Green color, thickness = 2

                    if (region.size()<30)
                        continue;
                    std::vector<cv::KeyPoint> keypoints1;
                    cv::Point2f center(0, 0);
                    for (const auto& point : region) {
                        cv::KeyPoint kp;
                        kp.size=10;
                        kp.pt = static_cast<cv::Point2f>(point);
                        keypoints1.push_back(kp);
                    }
    ;
                    // Сортировка KeyPoint по углам относительно центра
                    keypoints1=getBasePoint(keypoints1);

                    // Определение желаемого количества точек
                    int desiredPoints = static_cast<int>(max(coef, (long double)(1.2/keypoints1.size())) * keypoints1.size()); // Например, возьмем половину точек
                    //int desiredPoints=keypoints1.size()/2;
                    // Выбор пропорционального количества точек
                    std::vector<cv::KeyPoint> selectedKeypoints(keypoints1.begin(), keypoints1.begin() + desiredPoints);
                    keypoints.insert(keypoints.end(), selectedKeypoints.begin(), selectedKeypoints.end());

                }
                imshow("MSER CONTUR", resultImage2);
                //std::sort(keypoints.begin(), keypoints.end(), compareKeyPointsByResponse);
                std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                    return a.response<b.response;
                });
                //keypoints.resize(3000);
                Mat descriptor;
                //sift->detectAndCompute(tempo, tempo, keypoints, descriptor);
                //sift->compute(tempo, keypoints, descriptor);
                cv::normalize(descriptor, descriptor, 1.0, 0.0, cv::NORM_L2);
                //cout << keypoints.size() << " " << regions.size() << "\n"; cout.flush();
                cv::Mat resultImage;
                cv::cvtColor(tempo, resultImage, cv::COLOR_GRAY2BGR);

                //for (const auto& box : mserBoundBoxes) {
                //    cv::rectangle(resultImage, box, cv::Scalar(0, 255, 0), 2);
                //}

                // Визуализация KeyPoints
                //cv::drawKeypoints(resultImage, keypoints, resultImage, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
                resultImage.copyTo(images[i]);
                // Отобразить результат

                //equalizeHist(images[i], tempo);
                //std::vector<cv::KeyPoint> keypoints1;
                //Mat descriptors1;
                //sift->detect(tempo, keypoints1);
                //cv::drawKeypoints(resultImage, keypoints1, resultImage, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

                cv::imshow("MSER", resultImage);
                //cv::imshow("MSER and BRISK(min_marign="+to_string(mser->getMinMargin()), resultImage);
            //}

            cv::waitKey(0);
            //continue;
*/
            stringstream ss;
            ss << labels[i];
            //cv::FileStorage fs("Norm_Face/Descript_"+ss.str()+".yaml", cv::FileStorage::WRITE);
            //fs << "descriptors" << descriptor;
            //fs.release();
            //cv::FileStorage fsk("Norm_Face/KeyPoints_"+ss.str()+".yaml", cv::FileStorage::WRITE);
            //fsk << "keypoints" << keypoints;
            //fsk.release();
            imwrite("Norm_Face/Face_"+ss.str()+".jpg", images[i]);
            RE[i]=ss.str();

            //return ss.str();
        //});

    }
    for (int i=0; i<images.size(); i++){
        //RE[i].wait();
        //string ans = RE[i].get();
        string ans = RE[i];
        if (ans!=""){
            cout << ans << "\n";
            ouf << "Norm_Face/Face_"+ans+".jpg "+ans << "\n";
        }
    }
    model = LBPHFaceRecognizer::create(2,8,4,4);
    model->train(images, labels);
    model->write("LBPH.yaml");
}

bool MyFaceRecognition::Init(string FileData, string loadPath)
{
    net=cv::dnn::readNetFromCaffe("/home/nikita/Загрузки/deploy.prototxt", "/home/nikita/Загрузки/res10_300x300_ssd_iter_140000_fp16.caffemodel");
    facemark->loadModel("/home/nikita/QtProj/build-TrainLBF-Desktop-Release/MyModel3.yaml");
    //facemark->loadModel("/home/nikita/GSOC2017/data/lbfmodel.yaml");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    //return true;
    string fn_csv = string(FileData);
    try {
        read_csv(fn_csv, images, labels);
        descriptors.resize(labels.size());
        keypoints.resize(labels.size());
        for(int i=0; i<labels.size(); i++){
            stringstream ss; ss << labels[i];
            cv::FileStorage fs2("Norm_Face/Descript_"+ss.str()+".yaml", cv::FileStorage::READ);
            fs2["descriptors"] >> descriptors[i];
            fs2.release();
            cv::FileStorage fs3("Norm_Face/KeyPoints_"+ss.str()+".yaml", cv::FileStorage::READ);
            fs3["keypoints"] >> keypoints[i];
            fs3.release();
        }
    } catch (const cv::Exception& e) {
        cerr << "Ошибка открытия файла \"" << fn_csv << "\". Причина: " << e.msg << endl;
        exit(1);
    }
    if(images.size() <= 1) {
        string error_message = "Недостаточно изображений в базе!";
        CV_Error(Error::StsError, error_message);
    }
    model = LBPHFaceRecognizer::create(2,8,4,4);
    if (loadPath!="")
        model->read(loadPath);
        //model->load<LBPHFaceRecognizer>(loadPath);
    else
        model->train(images, labels);
    return true;
}


bool MyFaceRecognition::ExtractFace(Mat input, Mat& output, double ctotal, uint c){
    double total=ctotal;
    double marign=0.10;
    Mat image;
    input.copyTo(image);
    resize(image, image, Size(300,300), 0,0, INTER_CUBIC);
    //imshow("I", image);
    Mat inputBlob = cv::dnn::blobFromImage(image, 1.4);

    net.setInput(inputBlob);
    Mat detection = net.forward();
    int x1=0,y1=0,x2=0,y2=0;
    vector<Rect> faces;
    Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
    for (int i=0; i<detectionMat.rows; i++){
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > 0.7){
            x1 = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
            y1 = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
            x2 = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
            y2 = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);

            //imshow("input", image);
            auto w=x2-x1, h=y2-y1;
            auto x_marign=w*marign, y_marign=h*marign;
            x1-=x_marign; x2+=x_marign;
            y1-=y_marign; y2+=y_marign;

            if (faces.empty() || (faces[0].width<x2-x1 && faces[0].height<y2-y1))
                faces = {Rect(Point(x1,y1), Point(x2,y2))};

        }
    }
    vector<vector<Point2f> > shapes;
    double angle=10;
    //total=0.0;
    if (faces.empty() || faces.size()==0){
        auto M = getRotationMatrix2D(Point(input.cols/2,input.rows/2), angle, 1.0);
        input.copyTo(image);
        warpAffine(image, image, M, Size(image.cols, image.rows));
        total+=angle;
        if (total>=120) return false;
        return ExtractFace(image, output, total, c);
    }



    image=input;
    faces[0].x*=input.cols/300.0; faces[0].width*=input.cols/300.0;
    faces[0].y*=input.rows/300.0; faces[0].height*=input.rows/300.0;


    x1=faces[0].x; x2=faces[0].x+faces[0].width;
    y1=faces[0].y; y2=faces[0].y+faces[0].height;

    if(!facemark->fit(image, faces, shapes)){
        return false;
        auto M = getRotationMatrix2D(Point(x1+(x2-x1)/2,y1+(y2-y1)/2), angle, 1.0);
        warpAffine(image, image, M, Size(image.cols, image.rows));
        total+=angle;
        if (total>=120) return false;
        return ExtractFace(image, output, total, c);
    }
    //Успешно довращались
    auto deltaX=shapes[0][45].x-shapes[0][36].x, deltaY=shapes[0][45].y-shapes[0][36].y;
    //line(image, shapes[0][45], shapes[0][36], Scalar(255, 255, 0));
    Point centre(x1+(x2-x1)/2,y1+(y2-y1)/2);
    //drawMarker(image, centre, Scalar(255,0,0));
    double addi= 0;
    auto M = getRotationMatrix2D(centre, (atan(deltaY/deltaX)+addi)*180.0/3.14, 1.0);
    warpAffine(image, image, M, Size(image.cols, image.rows));
    total=total*3.14/180;
    total+=atan(deltaY/deltaX);
    if (ctotal-total*180/3.14>10){
        //cout << "02" << " ";
        return ExtractFace(image, output, total*180/3.14, c);
    }

    //drawFacemarks(image, shapes[0]);

    /*
    vector<Point2f> inp{
        shapes[0][36],
        shapes[0][45],
        shapes[0][54],
        shapes[0][48],
    };

    vector<Point2f> out{
        Point2f(image.cols*0.5-0.15*image.cols, image.rows*0.27),
        Point2f(image.cols*0.5+0.15*image.cols, image.rows*0.27),
        Point2f(image.cols*0.5+0.11*image.cols, image.rows*0.75),
        Point2f(image.cols*0.5-0.11*image.cols, image.rows*0.75)
    };

    Mat trans=getPerspectiveTransform(inp, out);
    warpPerspective(image, image, trans, Size(image.cols, image.rows), INTER_CUBIC);
    */

    //for (auto now: shapes[0]){
    //    drawMarker(image, now, Scalar(255,0,255));
    //}
    auto l=(sqrt(pow(x2-x1,2)+pow(y2-y1,2)))/2;
    auto a=acos((x1-centre.x)/l);
    auto x1t=l*cos(total+a), y1t=l*sin(total+a);
    a=acos((x2-centre.x)/l);
    auto x2t=l*cos(total+a), y2t=l*sin(total+a);
    auto x=max(abs(x1t), abs(x2t)), y=max(abs(y1t), abs(y2t));

    rectangle(image, Rect(centre.x-x, centre.y-y, 2*x, 2*y), Scalar(0,0,255), 5);
    //cout << Rect(centre.x-x, centre.y-y, 2*x, 2*y) << "\n";
    //imshow(image)
    //imshow("face", image);
    Rect rect(Point(max<double>(0, centre.x-x), max<double>(0,centre.y-y)),
              Point(min<double>(image.cols, centre.x+x), min<double>(image.rows,centre.y+y)));

    //Mat mask(image.size(), CV_8UC1, Scalar(0));
    //Mat bgdModel, fgdModel;
    //grabCut(image, mask, rect, bgdModel, fgdModel, 100, GC_INIT_WITH_RECT);
    //Mat foreground;
    //image.copyTo(foreground, (mask == GC_PR_FGD) | (mask == GC_FGD));
    //imshow("GrubCut", foreground);
    output=Mat(image, rect);

    //imshow("marker", output);
    vector<Point2f> shape;
    for(auto p : shapes[0]){
        shape.push_back(Point2f(
            (p.x-faces[0].x)*512.0/faces[0].width,
            (p.y-faces[0].y)*512.0/faces[0].height
        ));
    }
    NormalizeImage(output, output);
    getOnlyFace(output, output, shape, faces[0]);
    //resize(image, image, Size(300,300))

    //output=Mat(image, faces[0]);
    return true;
}



vector<pair<int, double> > MyFaceRecognition::GetSimilarFacesLBPH(Mat& ProcessingImage, int count, double tresh, uint n){
    int predictedLabel = -1;
    //GetNormalizateImage(ProcessingImage, ProcessingImage);
    //cvtColor(ProcessingImage, ProcessingImage, COLOR_RGB2GRAY);
    //normalize(ProcessingImage, ProcessingImage, 0, 255, NORM_MINMAX);
    //cvtColor(Image2, Image2, COLOR_RGB2GRAY);
    resize(ProcessingImage, ProcessingImage, Size(512,512), 0,0, INTER_CUBIC);
    //equalizeHist(ProcessingImage, ProcessingImage);
    //resize(Image2, Image2, Size(512, 512));
    Ptr<StandardCollector> PredictCollect = StandardCollector::create(tresh);
    model->predict(ProcessingImage, PredictCollect);
    auto res = PredictCollect->getResults(true);

    res.resize(min<int>(count, res.size()));
    //return res;
    Mat Result;
    vector<pair<double, Mat> > Rvec;
    std::vector<cv::KeyPoint> keypoints2;
    Mat descriptors1;
    int MSER_delta=5, MSER_min_area=60, MSER_max_area=14400; double MSER_max_variation=0.25, MSER_min_diversity=0.200000000001;
    int MSER_max_evolution=200; double MSER_area_treshhold=1.01, MSER_min_marign=0.0030000001; int MSER_edge_blur_stage=5;

    MSER_delta=1; MSER_min_area=200; MSER_max_area=3000;
    MSER_max_variation=0.8; MSER_min_diversity=0.1;
    MSER_area_treshhold=2; MSER_edge_blur_stage=18;
    MSER_max_evolution=1000; MSER_min_marign=0.01;


    int BRISK_tresh=30, BRISK_octaves=3; float BRISK_patternScale=1.0f;

     BRISK_tresh=30; BRISK_octaves=3;

    Ptr<MSER> mser = MSER::create(MSER_delta, MSER_min_area, MSER_max_area, MSER_max_variation, MSER_min_diversity, MSER_max_evolution, MSER_area_treshhold, MSER_min_marign, MSER_edge_blur_stage);

    Ptr<BRISK> brisk = BRISK::create(BRISK_tresh, BRISK_octaves, BRISK_patternScale);
    vector<vector<Point>> regions;
    vector<Rect> mserBoundBoxes;
    mser->detectRegions(ProcessingImage, regions, mserBoundBoxes);
    long double sum=0.0;
    for (const auto& region : regions){
        sum+=region.size();
    }
    long double coef=380/sum;
    for (const auto& region : regions){
        if (region.size()<30)
            continue;
        std::vector<cv::KeyPoint> keypoints1;
        for (const auto& point : region) {
            cv::KeyPoint kp;
            kp.size=10;
            kp.pt = static_cast<cv::Point2f>(point);
            keypoints1.push_back(kp);
        }

        keypoints1=getBasePoint(keypoints1);

        // Определение желаемого количества точек
        int desiredPoints = static_cast<int>(max(coef, (long double)(1.2/keypoints1.size())) * keypoints1.size()); // Например, возьмем половину точек

        // Выбор пропорционального количества точек
        std::vector<cv::KeyPoint> selectedKeypoints(keypoints1.begin(), keypoints1.begin() + desiredPoints);
        keypoints2.insert(keypoints2.end(), selectedKeypoints.begin(), selectedKeypoints.end());

    }
    //std::sort(keypoints.begin(), keypoints.end(), compareKeyPointsByResponse);
    std::sort(keypoints2.begin(), keypoints2.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
        return a.response<b.response;
    });
    keypoints2.resize(300);
    Ptr<SIFT> sift = SIFT::create(300, 4, 0.1, 15);
    //Ptr<AKAZE> akaze = AKAZE::create();

    Mat descriptor;
    sift->compute(ProcessingImage, keypoints2, descriptor);
    cv::normalize(descriptor, descriptor, 1.0, 0.0, cv::NORM_L2);
    //akaze->detectAndCompute(ProcessingImage, Mat(), keypoints2, descriptor);

    imwrite("MyWeb_"+to_string(n)+".jpg", ProcessingImage);
    cv::FileStorage fs("MyWebDescript_"+to_string(n)+".yaml", cv::FileStorage::WRITE);
    fs << "descriptors" << descriptor;
    fs.release();
    cv::FileStorage fsk("MyWebKeyPoints_"+to_string(n)+".yaml", cv::FileStorage::WRITE);
    fsk << "keypoints" << keypoints2;
    fsk.release();
    return res;
/*
    cv::drawKeypoints(ProcessingImage, keypoints2, ProcessingImage, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
    cv::imshow("MSER and BRISK", ProcessingImage);
    // Отобразить результат

    //equalizeHist(images[i], tempo);
    //std::vector<cv::KeyPoint> keypoints1;
    //Mat descriptors1;
    //sift->detect(tempo, keypoints1);
    //cv::drawKeypoints(resultImage, keypoints1, resultImage, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

    //cv::imshow("MSER and BRISK", resultImage);
    return {};
*/

    /*
    Ptr<SIFT> sift = SIFT::create();
    sift->setNFeatures(100);
    Mat tempo;
    equalizeHist(ProcessingImage, tempo);
    sift->detectAndCompute(tempo, Mat(), keypoints1, descriptors1);
    //vector<cv::Point> elasticGraph1 = createElasticGraphSIFT(keypoints1, descriptors1, ProcessingImage);
    */
    //Ptr<ORB> orb = ORB::create();
    //orb->detect(ProcessingImage, keypoints1);
    //vector<cv::Point> elasticGraph1 = createElasticGraphORB(keypoints1, ProcessingImage);
    future<pair<bool, double> > SIFTRes[min<int>(count, res.size())];
    for(int i=0; i<min<int>(count, res.size()); i++){

        SIFTRes[i] = std::async(std::launch::async, compareFacesWithSIFT_RANSAC, descriptor, getDescriptByLabel(res[i].first), keypoints2, getKeyPointByLabel(res[i].first), res[i].second);

    }
    double MyDist[2]={-1, 1000};
    double OtherDist[2]={-1, 1000};
    for(int i=0; i<min<int>(count, res.size()); i++){
        pair<bool, double> tt=SIFTRes[i].get();
            //;tt = compareFacesWithBRISK(descriptor, getDescriptByLabel(res[i].first), res[i].second);

        if (tt.first){
            if (res[i].first==65782 || res[i].first==3 || res[i].first==1 || res[i].first==2 || res[i].first==4 || res[i].first==5){
                MyDist[0]=max<double>(MyDist[0], tt.second);
                MyDist[1]=min<double>(MyDist[1], tt.second);
            }else{
                OtherDist[0]=max<double>(OtherDist[0], tt.second);
                OtherDist[1]=min<double>(OtherDist[1], tt.second);
            }
        }
        //if (tt.first){
        //    cout << res[i].first << " dist: " << tt.second << "\n";
        //}
    }

    if (MyDist[0]!=-1)
        cout << MyDist[0] << " ... " << MyDist[1] << "    MY\n";
    if (OtherDist[0]!=-1)
        cout << OtherDist[0] << " ... " << OtherDist[1] << "     OTHER\n\n";
    cout.flush();

    //drawFacemarks(ProcessingImage, elasticGraph1, Scalar(255));
    sort(Rvec.begin(), Rvec.end(), [](pair<double, Mat>& p1, pair<double, Mat>& p2){return p1.first>p2.first;});

    if (Rvec.size()){
        Result=Rvec[0].second;
        for (int q=1; q<Rvec.size(); q++)
            hconcat(Result, Rvec[q].second, Result);
        imshow("Pretendents", Result);
    }

    return res;
}

Mat Derivative(Mat src, int dx, int dy){
    Mat preparedSrc;
    src.copyTo(preparedSrc);
    //cvtColor(src, preparedSrc, COLOR_BGR2GRAY);
    preparedSrc.convertTo(preparedSrc, CV_32F, 2.5 / 255);
    if (preparedSrc.empty()) {
        throw runtime_error("Error");
    }
    Mat kernelRows, kernelColumns;
    getDerivKernels(kernelRows, kernelColumns, dx, dy, 3, true);
    float kernelFactor=2.0f;
    Mat multipliedKernelRows = kernelRows.mul(kernelFactor);
    Mat multipliedKernelColumns = kernelColumns.mul(kernelFactor);
    // Проверка размеров и типов ядер

    Mat ans;
    sepFilter2D(preparedSrc, ans, CV_32F,
                multipliedKernelRows,
                multipliedKernelColumns
    );
    return ans;
}

cv::Mat grayworld_normalization(const cv::Mat& image) {
    // Вычисление среднего значения по каналам
    cv::Scalar mean_value = cv::mean(image);

    // Вычисление коэффициентов коррекции
    double correction_factor = 110 / 128.0; // Берем среднее значение зеленого канала за основу

    // Применение коррекции к каждому каналу
    cv::Mat normalized_image = image * (correction_factor / 255.0);

    // Ограничение значений до диапазона [0, 255]
    cv::normalize(normalized_image, normalized_image, 0, 255, cv::NORM_MINMAX);

    return normalized_image;
}

void MyFaceRecognition::NormalizeImage(Mat& input, Mat& output){
    input.copyTo(output);

    resize(output, output, Size(512,512), 0,0, INTER_CUBIC);

    cv::Mat grayImage, grays[3];
    cv::cvtColor(output, grayImage, cv::COLOR_RGB2HSV);
    split(grayImage, grays);
    grayImage=grays[2];

    cv::normalize(grayImage, grayImage, 0, 255, cv::NORM_MINMAX);
    cv::medianBlur(grayImage, grayImage, 5);
    //cv::GaussianBlur(grayImage, grayImage, cv::Size(9, 9), 5, 5);
    Mat output2;
    output.copyTo(output2);
    output=grayImage;
}

void MyFaceRecognition::getOnlyFace(Mat& input, Mat& output3, vector<Point2f>  &shapes, Rect faceRec){
    Mat output2, output;
    input.copyTo(output);
    vector<vector<Point2f>> sh;
    vector<Rect> fa={Rect(0, 0, output.cols, output.rows)};
    facemark->fit(output, fa, sh);


    //if (sh.size())
    {
        Mat facemark; output.copyTo(facemark);
        cvtColor(facemark, facemark, COLOR_GRAY2RGB);
        drawFacemarks(facemark, sh[0]);
        //resize(facemark, facemark, Size(512,512));
        imshow("facemarks", facemark);
    }
    //return;
    //cvtColor(output, output, COLOR_RGB2GRAY);


    //morphologyEx(output, output, MORPH_CLOSE,
    //             getStructuringElement(MorphShapes::MORPH_ELLIPSE, Size(5,5)), Point(2,2), 1);

    /*

    Mat gradX=Derivative(output, 1, 0);

    Mat gradY=Derivative(output, 0, 1);
    magnitude(gradX, gradY, output);


    //imshow("prepare", output);
    output*=1.25f;


    morphologyEx(output, output, MORPH_DILATE,
                 getStructuringElement(MorphShapes::MORPH_ELLIPSE, Size(7,7)), Point(3,3), 1);

*/
    output = Mat::zeros(Size(output.cols, output.rows), CV_8UC1);
    output+=100;
    const float FloodFillTolerance = 0.06;
    Rect rect;

    for (int c=1; c<=16; ++c){
        line(output, sh[0][c-1], sh[0][c], Scalar(255,0,0), 3);
    }
    circle(output, sh[0][0]+(sh[0][16]-sh[0][0])/2, sqrt(pow(sh[0][0].x-sh[0][16].x, 2)+pow(sh[0][0].y-sh[0][16].y, 2))/2, Scalar(255), 3);
    /*
    line(output, sh[0][0], Point2f(Interest.x, Interest.y+Interest.height), Scalar(255), 3);
    line(output, sh[0][16], Point2f(Interest.x+Interest.width, Interest.y+Interest.height), Scalar(255), 3);
    line(output, Point2f(Interest.x, Interest.y), Point2f(Interest.x, Interest.y+Interest.height), Scalar(255), 3);
    line(output, Point2f(Interest.x+Interest.width, Interest.y), Point2f(Interest.x+Interest.width, Interest.y+Interest.height), Scalar(255), 3);
    line(output, Point2f(Interest.x, Interest.y), Point2f(Interest.x+Interest.width, Interest.y), Scalar(255), 3);
*/
    //findContours()
    vector<Point2f> seedPoints={
        Point2f(0, 0),
        Point2f(output.cols-1, 0),
        (sh[0][4]-sh[0][33])*1.3+sh[0][33],
        (sh[0][12]-sh[0][33])*1.3+sh[0][33],
    };
    for(auto seedPoint : seedPoints){
        try{
        floodFill(output,
                  seedPoint,
                  Scalar(0),
                  nullptr,
                  Scalar(FloodFillTolerance),
                  Scalar(FloodFillTolerance)
        );
        }catch(...){

        }
    }


/*
    morphologyEx(output, output, MORPH_DILATE,
                 getStructuringElement(MorphShapes::MORPH_CROSS, Size(3,3)), Point(1,1), 1);

    morphologyEx(output, output, MORPH_CLOSE,
                 getStructuringElement(MorphShapes::MORPH_CROSS, Size(3,3)), Point(1,1), 2);

    morphologyEx(output, output, MORPH_DILATE,
                 getStructuringElement(MorphShapes::MORPH_CROSS, Size(3,3)), Point(1,1), 1);
*/
    //imshow("Begore trash", output);

    threshold(
        output,
        output,
        0,
        255,
        THRESH_TOZERO
    );
    output.convertTo(output, CV_8UC1, 255);
    output2=Mat::zeros(input.rows, input.cols, CV_8UC1);
    //imshow("i");
    imshow("Trashhold", output);
    input.copyTo(output2, output);

    output2.copyTo(output);
    //grayworld_normalization(output);
    normalize(output, output, 0, 255, NORM_MINMAX);
    //cvtColor(output, output, COLOR_RGB2GRAY);

    /*
    morphologyEx(output, output, MORPH_CLOSE,
                 getStructuringElement(MorphShapes::MORPH_CROSS, Size(3,3)), Point(1,1), 2);
`   */
    resize(output, output, Size(512,512), 0,0, INTER_CUBIC);
    //drawMarker(output, (shapes[4]-shapes[33])*1.3+shapes[33], Scalar(255,0,0));
    //drawMarker(output, (shapes[12]-shapes[33])*1.3+shapes[33], Scalar(255,0,0));
    //cvtColor(output, output, COLOR_BGR2RGB);
    //Mat t;
    //output2.copyTo(t, output);
    //output=t;
    // Everything non-filled becomes white

    //cvtColor(output, output, COLOR_RGB2GRAY);
    //normalize(output, output, 0, 255, NORM_MINMAX);

    //output.convertTo(output, CV_32F, 1.0/255)
    //GaussianBlur(output, output, Size(5, 5), 0);
    //medianBlur(output, output, 5);
    /*
    cvtColor(input, input, COLOR_BGR2Lab);
    Mat chanels[input.channels()];
    split(output, chanels);
    for (auto now : chanels){
        auto CLAHE = createCLAHE(3, Size(8,8));
        CLAHE->apply(now, now);
    }
    merge(chanels, input.channels(), output);

    cvtColor(input, input, COLOR_Lab2BGR);*/
    //output2.copyTo(output);
    //if (c>1 || !ExtractFace(output, output,0.0, c+1))
    //    output2.copyTo(output);

    //morphologyEx(output, output, MORPH_DILATE,
    //             getStructuringElement(MorphShapes::MORPH_CROSS, Size(3,3)), Point(2,2), 1);
    //output2 = Mat
    //cvtColor(output, output2, COLOR_RGB2HSV);
    //split(output2, chanel);
    //output=chanel[0];
    output.copyTo(output3);
}

pair<bool, double> MyFaceRecognition::compareFacesWithSIFT_RANSAC(cv::Mat descriptorsReference, cv::Mat descriptorsTest, vector<KeyPoint> kp1, vector<KeyPoint> kp2, double adding) {
    if (descriptorsReference.empty() || descriptorsTest.empty() || descriptorsReference.size() != descriptorsTest.size() || descriptorsReference.type() != descriptorsTest.type()) {
        cout << descriptorsReference.empty() << " " << descriptorsTest.empty() << " " << (descriptorsReference.size() != descriptorsTest.size()) << " " << (descriptorsReference.type() != descriptorsTest.type()) << "\n";
        return {false, 0};
    }

    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptorsReference, descriptorsTest, matches);
    sort(matches.begin(), matches.end(), [](cv::DMatch p1, cv::DMatch p2){return p1.distance > p2.distance;});
    matches.resize(100);
    std::vector<cv::Point2f> matchedPointsReference;
    std::vector<cv::Point2f> matchedPointsTest;
    for (const cv::DMatch& match : matches) {
        matchedPointsReference.push_back(kp1[match.queryIdx].pt);
        matchedPointsTest.push_back(kp2[match.trainIdx].pt);
    }
    cv::Mat homography = cv::findHomography(matchedPointsReference, matchedPointsTest, cv::RANSAC);
    std::vector<cv::DMatch> filteredMatches;
    long double sum_mist=0.0;
    for (size_t i = 0; i < matches.size(); ++i) {
        cv::Mat pointReference = (cv::Mat_<double>(3, 1) << matchedPointsReference[i].x, matchedPointsReference[i].y, 1.0);
        cv::Mat transformedPoint = homography * pointReference;
        double dx = transformedPoint.at<double>(0, 0) - matchedPointsTest[i].x;
        double dy = transformedPoint.at<double>(1, 0) - matchedPointsTest[i].y;
        double distance = std::sqrt(dx*dx + dy*dy);
        if (distance < 40) {
            filteredMatches.push_back(matches[i]);
            sum_mist+=matches[i].distance;
        }
    }
    matches=filteredMatches;
    sum_mist/=matches.size();
    double totalDistance = 0.0;
    for (const cv::DMatch& match : matches) {
        totalDistance += match.distance;
    }
    double avgDistance = totalDistance / matches.size();

    double similarityThreshold = 50000;

    // Сравнение среднего расстояния с порогом
    if (avgDistance < similarityThreshold || true) {
        return {true, avgDistance};
    }
    /*if (cosineDistance > 0.7) {
        return {true, cosineDistance};
    }*/
    return {false, 0};
}
