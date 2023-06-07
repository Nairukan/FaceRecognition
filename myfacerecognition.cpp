#include "myfacerecognition.h"

#include <future>

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
    net=cv::dnn::readNetFromCaffe("/home/nikita/Загрузки/deploy.prototxt", "/home/nikita/Загрузки/res10_300x300_ssd_iter_140000_fp16.caffemodel");
    facemark->loadModel("/home/nikita/GSOC2017/data/lbfmodel.yaml");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    string fn_csv = string(FileData);
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
    Ptr<SIFT> sift = SIFT::create();
    sift->setNFeatures(100);
    ofstream ouf ("Norm_Face/Faces.csv");
    //future<string> RE[images.size()];
    string RE[images.size()];
    for (int i=0; i<images.size(); i++){
        NormalizeImage(images[i], images[i]);
        if (!this->ExtractFace(images[i], images[i])){
            continue;
        }
        cout << i << "\n";
        //RE[i] = std::async(std::launch::async, [=]{
            std::vector<cv::KeyPoint> keypoints1;
            Mat descriptors1;

            resize(images[i], images[i], Size(512, 512));
            some(images[i], images[i]);
            Mat tempo;
            equalizeHist(images[i], tempo);
            sift->detectAndCompute(tempo, Mat(), keypoints1, descriptors1);
            stringstream ss;
            ss << labels[i];
            cv::FileStorage fs("Norm_Face/Descript_"+ss.str()+".yaml", cv::FileStorage::WRITE);
            fs << "descriptors" << descriptors1;
            fs.release();
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
    facemark->loadModel("/home/nikita/GSOC2017/data/lbfmodel.yaml");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    string fn_csv = string(FileData);
    try {
        read_csv(fn_csv, images, labels);
        descriptors.resize(labels.size());
        for(int i=0; i<labels.size(); i++){
            stringstream ss; ss << labels[i];
            cv::FileStorage fs2("Norm_Face/Descript_"+ss.str()+".yaml", cv::FileStorage::READ);
            fs2["descriptors"] >> descriptors[i];
            fs2.release();
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


bool MyFaceRecognition::ExtractFace(Mat input, Mat& output, double ctotal){
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
            //rectangle(image, faces[0], Scalar(255,0,255));
        }
    }
    vector<vector<Point2f> > shapes;
    double angle=20;
    //total=0.0;
    if (faces.empty() || faces.size()==0){
        auto M = getRotationMatrix2D(Point(image.cols/2,image.rows/2), angle, 1.0);
        warpAffine(image, image, M, Size(image.cols, image.rows));
        total+=angle;
        if (total>=360) return false;
        return ExtractFace(image, output, total);
    }

    x1=faces[0].x; x2=faces[0].x+faces[0].width;
    y1=faces[0].y; y2=faces[0].y+faces[0].height;
    if(!facemark->fit(image, faces, shapes)){
        auto M = getRotationMatrix2D(Point(x1+(x2-x1)/2,y1+(y2-y1)/2), angle, 1.0);
        warpAffine(image, image, M, Size(image.cols, image.rows));
        total+=angle;
        if (total>=360) return false;
        return ExtractFace(image, output, total);
    }
    //Успешно довращались
    auto deltaX=shapes[0][45].x-shapes[0][36].x, deltaY=shapes[0][45].y-shapes[0][36].y;
    //line(image, shapes[0][45], shapes[0][36], Scalar(255, 255, 0));
    Point centre(x1+(x2-x1)/2,y1+(y2-y1)/2);
    //drawMarker(image, centre, Scalar(255,0,0));
    auto M = getRotationMatrix2D(centre, atan(deltaY/deltaX)*180.0/3.14, 1.0);
    warpAffine(image, image, M, Size(image.cols, image.rows));
    total=total*3.14/180;
    total+=atan(deltaY/deltaX);
    if (ctotal-total*180/3.14>10){
        //cout << "02" << " ";
        return ExtractFace(image, output, total*180/3.14);
    }
    auto l=(sqrt(pow(x2-x1,2)+pow(y2-y1,2)))/2;
    auto a=acos((x1-centre.x)/l);
    auto x1t=l*cos(total+a), y1t=l*sin(total+a);
    a=acos((x2-centre.x)/l);
    auto x2t=l*cos(total+a), y2t=l*sin(total+a);
    auto x=max(abs(x1t), abs(x2t)), y=max(abs(y1t), abs(y2t));

    //rectangle(image, Rect(centre.x-x, centre.y-y, 2*x, 2*y), Scalar(255,0,255));
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

    //output=Mat(image, faces[0]);
    return true;
}



vector<pair<int, double> > MyFaceRecognition::GetSimilarFacesLBPH(Mat& ProcessingImage, int count, double tresh){
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

    Mat Result;
    vector<pair<double, Mat> > Rvec;
    std::vector<cv::KeyPoint> keypoints1;
    Mat descriptors1;
    Ptr<SIFT> sift = SIFT::create();
    sift->setNFeatures(100);
    Mat tempo;
    equalizeHist(ProcessingImage, tempo);
    sift->detectAndCompute(tempo, Mat(), keypoints1, descriptors1);
    //vector<cv::Point> elasticGraph1 = createElasticGraphSIFT(keypoints1, descriptors1, ProcessingImage);

    //Ptr<ORB> orb = ORB::create();
    //orb->detect(ProcessingImage, keypoints1);
    //vector<cv::Point> elasticGraph1 = createElasticGraphORB(keypoints1, ProcessingImage);
    /*future<pair<bool, double> > SIFTRes[min<int>(count, res.size())];
    for(int i=0; i<min<int>(count, res.size()); i++){
        SIFTRes[i] = std::async(std::launch::async, [=]{return compareFacesWithSIFT(descriptors1, getDescriptByLabel(res[i].first), res[i].second);});

    }
    double MyDist[2]={-1, 1000};
    double OtherDist[2]={-1, 1000};
    for(int i=0; i<min<int>(count, res.size()); i++){
        SIFTRes[i].wait();
        pair<bool, double> tt=SIFTRes[i].get();
        if (tt.first){
            if (res[i].first==65782 || res[i].first==3 || res[i].first==1 || res[i].first==2){
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


    //drawFacemarks(ProcessingImage, elasticGraph1, Scalar(255));
    sort(Rvec.begin(), Rvec.end(), [](pair<double, Mat>& p1, pair<double, Mat>& p2){return p1.first>p2.first;});

    if (Rvec.size()){
        Result=Rvec[0].second;
        for (int q=1; q<Rvec.size(); q++)
            hconcat(Result, Rvec[q].second, Result);
        imshow("Pretendents", Result);
    }
    */
    return res;
}


void MyFaceRecognition::NormalizeImage(Mat& input, Mat& output){
    /*input.copyTo(output);
    //resize(answer, answer, Size(300, 300));
    //GaussianBlur(output, output, Size(5, 5), 0);
    //medianBlur(output, output, 5);
    cvtColor(input, input, COLOR_BGR2Lab);
    Mat chanels[input.channels()];
    split(output, chanels);
    for (auto now : chanels){
        auto CLAHE = createCLAHE(3, Size(8,8));
        CLAHE->apply(now, now);
    }
    merge(chanels, input.channels(), output);

    cvtColor(input, input, COLOR_Lab2BGR);
    */
}

pair<bool, double> MyFaceRecognition::compareFacesWithSIFT(cv::Mat descriptorsReference, cv::Mat descriptorsTest, double adding) {
    //cv::normalize(descriptorsReference, descriptorsReference, cv::NORM_L2);
    //cv::normalize(descriptorsTest, descriptorsTest, cv::NORM_L2);
    //double cosineDistance = descriptorsReference.dot(descriptorsTest);


    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsReference, descriptorsTest, matches);
    sort(matches.begin(), matches.end(), [](DMatch p1, DMatch p2){return p1.distance>p2.distance;});
    matches.resize(50);
    double totalDistance = 0.0;
    for (const cv::DMatch& match : matches) {
        totalDistance += match.distance;
    }
    double avgDistance = totalDistance / matches.size();


/*
    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher->knnMatch(descriptorsReference, descriptorsTest, knnMatches, 2);
    const float ratioThreshold = 0.7f;
    std::vector<cv::DMatch> goodMatches;
    for (const auto& match : knnMatches) {
        if (match[0].distance < ratioThreshold * match[1].distance) {
            goodMatches.push_back(match[0]);
        }
    }
    double totalDistance = 0.0;
    for (const cv::DMatch& match : goodMatches) {
        totalDistance += match.distance;
    }
    double avgDistance = totalDistance / goodMatches.size();
*/

/*
    cv::BFMatcher matcher(cv::NORM_L2);

    // Сопоставление дескрипторов ключевых точек между эталоном и тестовым изображением
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsReference, descriptorsTest, matches);

    std::vector<cv::Point2f> matchedPointsReference;
    std::vector<cv::Point2f> matchedPointsTest;
    for (const cv::DMatch& match : matches) {
        matchedPointsReference.push_back(KPR[match.queryIdx].pt);
        matchedPointsTest.push_back(keypointsTest[match.trainIdx].pt);
    }
    cv::Mat homography = cv::findHomography(matchedPointsReference, matchedPointsTest, cv::RANSAC);
    std::vector<cv::DMatch> filteredMatches;
    for (size_t i = 0; i < matches.size(); ++i) {
        cv::Mat pointReference = cv::Mat(cv::Point3f(matchedPointsReference[i].x, matchedPointsReference[i].y, 1.0));
        cv::Mat transformedPoint = homography * pointReference;
        double dx = transformedPoint.at<double>(0, 0) - matchedPointsTest[i].x;
        double dy = transformedPoint.at<double>(1, 0) - matchedPointsTest[i].y;
        double distance = std::sqrt(dx*dx + dy*dy);
        if (distance < 10.0) {
            filteredMatches.push_back(matches[i]);
        }
    }
    double totalDistance = 0.0;
    for (const cv::DMatch& match : filteredMatches) {
        totalDistance += match.distance;
    }
    double avgDistance = totalDistance / filteredMatches.size();
    */

    // Установка порога для схожести
    //avgDistance+=adding;
    double similarityThreshold = 5088880.0;

    // Сравнение среднего расстояния с порогом
    if (avgDistance < similarityThreshold) {
        return {true, avgDistance};
    }
    /*if (cosineDistance > 0.7) {
        return {true, cosineDistance};
    }*/
    return {false, 0};
}

double MyFaceRecognition::ElasticGraphMath(vector<cv::Point> elasticGraph1, int label, Mat& matched, double coff){
    Mat image2; this->getMatByLabel(label).copyTo(image2);
    std::vector<cv::KeyPoint>  keypoints2;
    //Mat descriptors2;
    //Ptr<SIFT> sift = SIFT::create();
    //sift->detectAndCompute(image2, Mat(), keypoints2, descriptors2);
    Ptr<ORB> orb = ORB::create();
    //orb->detect(image, keypoints1);
    orb->detect(image2, keypoints2);

    // Создание эластических графов из ключевых точек
    //vector<cv::Point> elasticGraph1 = createElasticGraphORB(keypoints1, image);
    //vector<cv::Point> elasticGraph1; this->facemark->fit(image, vector<Rect>{Rect(0,0,image.cols, image.rows)}, elasticGraph1);
    vector<cv::Point> elasticGraph2 = createElasticGraphORB(keypoints2, image2);
    //vector<cv::Point> elasticGraph2 = createElasticGraphSIFT(keypoints2, descriptors2, image2);
    //vector<cv::Point> elasticGraph2; this->facemark->fit(image2, vector<Rect>{Rect(0,0,image2.cols, image2.rows)}, elasticGraph2);
    drawFacemarks(image2, elasticGraph2, Scalar(255));
    double similarity = compareElasticGraphs(elasticGraph1, elasticGraph2)*coff;
    if (similarity>=0.6){
        stringstream ss; ss<<similarity;
        putText(image2, ss.str(), Point(0,40), FONT_HERSHEY_COMPLEX, 0.6, Scalar(0), 2, LINE_AA);
        matched=image2;

    }
    return similarity;

}
