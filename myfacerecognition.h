#ifndef MYFACERECOGNITION_H
#define MYFACERECOGNITION_H

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <future>
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
    void getOnlyFace(Mat& input, Mat& output, vector<Point2f> &shapes, Rect faceRec=Rect(0,0,0,0));
    void NormalizeImage(Mat& input, Mat& output);
    void PrepareDataset(string FileData);
    bool ExtractFace(Mat input, Mat& output, double total=0.0, uint c=0);
    vector<pair<int, double> > GetSimilarFacesLBPH(Mat& ProcessingImage, int count, double tresh=30.0, uint n=0);
    double ElasticGraphMath(vector<cv::Point> elasticGraph1, int label, Mat& Mathed, double coff=1.0);
    static pair<bool, double> compareFacesWithSIFT_RANSAC(Mat, Mat, vector<KeyPoint>, vector<KeyPoint>, double=0.0);

    static Mat mySuperAlgo(vector<Point2f> shape, const double kHorizontal, const double kVertical, InputArray& image, int size, int steps=0){
        Point2f& ul(shape[0]), &ur(shape[1]), &dl(shape[2]), &dr(shape[3]);
        const vector<Point2f> p_dst={   Point2f(0, 0),
                                        Point2f(size, 0),
                                        Point2f(0, size),
                                        Point2f(size, size)
                                    };

        Mat im;
        if (!steps){
            auto matrix=getPerspectiveTransform(shape, p_dst, DECOMP_SVD);
            cv::warpPerspective(image, im, matrix, Size(size, size));
            return im;
        }
        Mat parted[4];

        const Point2f u=ur+(ul-ur)/(kHorizontal+1);
        const Point2f d=dr+(dl-dr)/(kHorizontal+1);
        const Point2f ml=ul+(dl-ul)/(kVertical+1), mr=ur+(dr-ur)/(kVertical+1);
        const Point2f m=mr+(ml-mr)/(kHorizontal+1);
        vector<vector<Point2f>> p_src={ {ul, u, ml, m},
                                        {ml, m, dl, d},
                                        {u, ur, m, mr},
                                        {m, mr, d, dr},
                                      };

        if (steps>=4){
            future<Mat> fut[4];
            for (int i=0; i<4; ++i){
                fut[i]=async(launch::async, mySuperAlgo, p_src[i], sqrt(kHorizontal), sqrt(kVertical), image, size/2, steps-1);
            }
            for (int i=0; i<4; ++i){
                parted[i]=fut[i].get();
            }
        }else{
            for (int i=0; i<4; ++i){
                    parted[i]=mySuperAlgo(p_src[i], sqrt(kHorizontal), sqrt(kVertical), image, size/2, steps-1);
            }
        }
       vconcat(parted[0], parted[1], parted[0]);
       vconcat(parted[2], parted[3], parted[1]);
       hconcat(parted[0], parted[1], im);
       return im;
    }

    Mat myUnperspectiveAlgo(vector<Point2f> shape, vector<double> Coefs, const Point2f pHl, const Point2f pHr, InputArray& image, int size, int steps=0){
        Point2f& ul(shape[0]), &ur(shape[1]), &um(shape[2]), &dl(shape[3]), &dr(shape[4]), &dm(shape[5]);


        const double tleft=Coefs[0]*Coefs[4]-Coefs[3]*Coefs[1];
        const double tright=Coefs[6]*Coefs[10]-Coefs[9]*Coefs[7];

        const double cleft=Coefs[2]*Coefs[4]-Coefs[5]*Coefs[1];
        const double cright=Coefs[8]*Coefs[10]-Coefs[11]*Coefs[7];

        const double kleft=(ul.x*tleft-cleft)/(um.x*tleft-cleft);
        const double kright=(um.x*tright-cright)/(ur.x*tright-cright);
        //cout << "kleft: " << kleft << endl << "kright: " << kright << endl;
        Mat parted[4];

        const Point2f u=um;
        const Point2f d=dm;
        const Point2f ml=(ul+dl)/2, mr=(ur+dr)/2;
        const Point2f m=(um+dm)/2;
        vector<vector<Point2f>> p_src={ {ul, u, ml, m},
                                        {ml, m, dl, d},
                                        {u, ur, m, mr},
                                        {m, mr, d, dr},
                                      };

        const vector<double> kHorizontal({kleft, kleft, kright, kright});
        if (steps>=4){
            future<Mat> fut[4];
            for (int i=0; i<4; ++i){
                fut[i]=async(launch::async, mySuperAlgo, p_src[i], kHorizontal[i], 1, image, size/2, steps-1);
            }
            for (int i=0; i<4; ++i){
                parted[i]=fut[i].get();
            }
        }else{
            for (int i=0; i<4; ++i){
                    parted[i]=mySuperAlgo(p_src[i], kHorizontal[i], 1, image, size/2, steps-1);
            }
        }
       vconcat(parted[0], parted[1], parted[0]);
       vconcat(parted[2], parted[3], parted[1]);
       Mat im;
       hconcat(parted[0], parted[1], im);
       return im;
    }

    template<class T>
    T PointAcross2Lines(T p11, T p12, T p21, T p22){
        float A1 = p12.y - p11.y;
        float B1 = p11.x - p12.x;
        float C1 = A1 * p11.x + B1 * p11.y;

        // Уравнение второй линии: A2x + B2y = C2
        float A2 = p22.y - p21.y;
        float B2 = p21.x - p22.x;
        float C2 = A2 * p21.x + B2 * p21.y;

        // Определитель системы уравнений
        float determinant = A1 * B2 - A2 * B1;
        if (determinant == 0) {
            return cv::Point2f(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
        } else {
            // Правило Крамера для решения системы уравнений
            float x = (B2 * C1 - B1 * C2) / determinant;
            float y = (A1 * C2 - A2 * C1) / determinant;
            return cv::Point2f(x, y);
        }
    }

    template<class T>
    Point2f PointAcross2Lines(T A1, T B1, T C1, T A2, T B2, T C2){

        // Определитель системы уравнений
        float determinant = A1 * B2 - A2 * B1;
        if (determinant == 0) {
            return cv::Point2f(std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
        } else {
            // Правило Крамера для решения системы уравнений
            float x = (B2 * C1 - B1 * C2) / determinant;
            float y = (A1 * C2 - A2 * C1) / determinant;
            return cv::Point2f(x, y);
        }
    }

    cv::Mat retinexMSR(const cv::Mat &src, const std::vector<int> &scales) {
        CV_Assert(src.channels() == 3);  // Проверка на цветное изображение

        cv::Mat srcLog;
        src.convertTo(srcLog, CV_64FC3);
        cv::Mat sumMat = cv::Mat::zeros(src.size(), CV_64FC3);

        vector<std::future<cv::Mat> >tem_res;
        for (size_t i = 0; i < scales.size(); ++i) {
            tem_res.emplace_back(async(launch::async, ([&srcLog](int ksize){
                cv::Mat imgBlur;
                cv::GaussianBlur(srcLog, imgBlur, cv::Size(0,0), ksize);
                Mat sumMat=srcLog - imgBlur;
                cv::normalize(sumMat, sumMat, 0, 255, cv::NORM_MINMAX);
                return sumMat;
            }), scales[i]
                                      )
                                 );
        }
        for (size_t i = 0; i < scales.size(); ++i) {

            sumMat += tem_res[i].get();
        }

        sumMat/=scales.size();
        cv::normalize(sumMat, sumMat, 0, 255, cv::NORM_MINMAX);
        sumMat.convertTo(sumMat, CV_8UC3);

        return sumMat;
    }


    Mat HorizontalPerpectiveNormalize(InputArray& inp, Rect rec=Rect()){
        Mat image; inp.copyTo(image);
        Mat input; inp.copyTo(input);
        if (rec==Rect()) rec=Rect(0, 0, image.cols, image.rows);
        vector<vector<Point2f> > shape;

        if(!facemark->fit(image, vector<Rect>{rec}, shape)){
            return Mat::zeros(1,1, CV_8UC1);
        }
        Mat output = Mat::zeros(Size(image.cols, image.rows), CV_8UC1);
        output+=100;
        const float FloodFillTolerance = 0.06;
        Rect rect;

        for (int c=1; c<=16; ++c){
            line(output, shape[0][c-1], shape[0][c], Scalar(255,0,0), 3);
        }
        /*
        line(output, shape[0][26], shape[0][16], Scalar(255,0,0), 3);
        line(output, shape[0][26], shape[0][25], Scalar(255,0,0), 3);
        line(output, shape[0][24], shape[0][25], Scalar(255,0,0), 3);

        line(output, shape[0][24], shape[0][19], Scalar(255,0,0), 3);

        line(output, shape[0][18], shape[0][19], Scalar(255,0,0), 3);
        line(output, shape[0][18], shape[0][17], Scalar(255,0,0), 3);
        line(output, shape[0][17], shape[0][0], Scalar(255,0,0), 3);
        */
        circle(output, shape[0][0]+(shape[0][16]-shape[0][0])/2, sqrt(pow(shape[0][0].x-shape[0][16].x, 2)+pow(shape[0][0].y-shape[0][16].y, 2))/2, Scalar(255), 3);

        vector<Point2f> seedPoints={
            Point2f(0, 0),
            Point2f(output.cols-1, 0),
            (shape[0][4]-shape[0][33])*1.3+shape[0][33],
            (shape[0][12]-shape[0][33])*1.3+shape[0][33],
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
        threshold(
            output,
            output,
            0,
            255,
            THRESH_TOZERO
        );
        output.convertTo(output, CV_8UC1, 255);
        auto mask=output;
        Mat temp_mat; input.copyTo(temp_mat, mask);
        temp_mat.copyTo(input);

        rectangle(image, rec, Scalar(255,255), 2);
        vector<Point2f> shapes=shape[0];
        double left=image.cols, right=0, up=image.rows, down=0;
        int left_ind=-1, right_ind=-1;
        for (int i=0; i<shapes.size(); ++i){
            double x=shapes[i].x, y=shapes[i].y;
            if (x<left){
                left=x;
                left_ind=i;
            }
            if (x>right){
                right=x;
                right_ind=i;
            }
            if (y<up){
                up=y;
            }
            if (y>down){
                down=y;
            }
        }
        drawFacemarks(image, shapes);
        rectangle(image, Rect(Point(left, up), Point(right, down)), Scalar(0, 255, 255), 2);

        Point2d d=shapes[8], dl(left, 0), dr(right, 0), ul(left, 0), ur(right, 0);
        Point2d u=shapes[24];
        Point2d p11=shapes[8], p12=shapes[7];
        Point2d p31=shapes[8], p32=shapes[9];


        Point2d p21=shapes[36], p22=shapes[39];
        Point2d p41=shapes[42], p42=shapes[45];
        Point2d p2m=(shapes[62]+shapes[66])/2;


        Point2d pHl=PointAcross2Lines(p11, p12, p21, p22);
        Point2d pHr=PointAcross2Lines(p31, p32, p41, p42);

        vector<Point2f> ps=shapes;
        vector<Point2f> ps_left, ps_right;
        for (auto p: ps){
            if (p.x>d.x) ps_right.push_back(p);
            else if(p.x<d.x) ps_left.push_back(p);
            else{
                ps_right.push_back(p);
                ps_left.push_back(p);
            }
        }
        vector<Point2f> buffer;
        buffer.reserve(shapes.size());
        double Aul, Bul, Cul;
        double Aur, Bur, Cur;
        double Adl, Bdl, Cdl;
        double Adr, Bdr, Cdr;

        //ps=ps_left;
        future<Point2f> pupl_f=async(launch::async, [&Aul, &Bul, &Cul, pHl](vector<Point2f> ps, Point2f last_point){
            vector<Point2f> buffer;
            do{
                //cout << "upl " << ps.size() << endl;
                Aul=pHl.y-last_point.y; Bul=last_point.x-pHl.x; Cul=Aul*last_point.x+Bul*last_point.y;
                buffer.clear();
                for (auto p: ps){
                    if (p.y-(Cul-Aul*p.x)/Bul<-0.1)
                        buffer.push_back(p);
                }
                ps=buffer;
                if (ps.size())
                    last_point=ps[0];
            }
            while(ps.size());
            return last_point;
        }, ps_left, shapes[19]);

        future<Point2f> pupr_f=async(launch::async, [&Aur, &Bur, &Cur, pHr](vector<Point2f> ps, Point2f last_point){
            vector<Point2f> buffer;
            do{
                //cout << "upl " << ps.size() << endl;
                Aur=pHr.y-last_point.y; Bur=last_point.x-pHr.x; Cur=Aur*last_point.x+Bur*last_point.y;
                buffer.clear();
                for (auto p: ps){
                    if (p.y-(Cur-Aur*p.x)/Bur<-0.1)
                        buffer.push_back(p);
                }
                ps=buffer;
                if (ps.size())
                    last_point=ps[0];
            }
            while(ps.size());
            return last_point;
        }, ps_right, shapes[24]);

        future<Point2f> pdownl_f=async(launch::async, [&Adl, &Bdl, &Cdl, pHl](vector<Point2f> ps, Point2f last_point){
            vector<Point2f> buffer;
            do{
                //cout << "downl " << ps.size() << endl;
                Adl=pHl.y-last_point.y; Bdl=last_point.x-pHl.x; Cdl=Adl*last_point.x+Bdl*last_point.y;
                buffer.clear();
                for (auto p: ps){
                    if ((Cdl-Adl*p.x)/Bdl-p.y<-0.1)
                        buffer.push_back(p);
                }
                ps=buffer;
                if (ps.size())
                    last_point=ps[0];
            }
            while(ps.size());
            return last_point;
        }, ps_left, d);

        future<Point2f> pdownr_f=async(launch::async, [&Adr, &Bdr, &Cdr, pHr](vector<Point2f> ps, Point2f last_point){
            vector<Point2f> buffer;
            do{
                //cout << "downl " << ps.size() << endl;
                Adr=pHr.y-last_point.y; Bdr=last_point.x-pHr.x; Cdr=Adr*last_point.x+Bdr*last_point.y;
                buffer.clear();
                for (auto p: ps){
                    if ((Cdr-Adr*p.x)/Bdr-p.y<-0.1)
                        buffer.push_back(p);
                }
                ps=buffer;
                if (ps.size())
                    last_point=ps[0];
            }
            while(ps.size());
            return last_point;
        }, ps_right, d);

        Point2f pupl=pupl_f.get();
        Point2f pdownl=pdownl_f.get();
        Point2f pupr=pupr_f.get();
        Point2f pdownr=pdownr_f.get();

        //cout << "calc start" << endl;


        ul=Point2d(left, (Cul-Aul*left)/Bul);
        ur=Point2d(right, (Cur-Aur*right)/Bur);
        Point2d um(d.x, min((Cul-Aul*d.x)/Bul, (Cur-Aur*d.x)/Bur));

        dl=Point2d(left, (Cdl-Adl*left)/Bdl);
        dr=Point2d(right, (Cdr-Adr*right)/Bdr);
        Point2d dm(d.x, max((Cdl-Adl*d.x)/Bdl, (Cdr-Adr*d.x)/Bdr));


        Point2f pV;
        //cout << "calc end" << endl;

        line(image, dl, dm, Scalar(0,255,0), 2);
        line(image, dl, ul, Scalar(0,255,0), 2);
        line(image, ul, um, Scalar(0,255,0), 2);
        line(image, dr, dm, Scalar(0,255,0), 2);
        line(image, dr, ur, Scalar(0,255,0), 2);
        line(image, ur, um, Scalar(0,255,0), 2);
        line(image, dm, um, Scalar(0,255,0), 2);

        //const double kHorizontal=FR.euqlideDistance(dl,d)/FR.euqlideDistance(dr,d);
        //const double kHorizontal=FR.euqlideDistance(dl,ul)/ FR.euqlideDistance(dr,ur);

        /*
        double maxInaccuracy=1;
        double best_ndr_x;
        double t=Au*Bd-Ad*Bu, c=Cd*Bu-Cu*Bd;
        double k=t*d.x-c;
        long double Al, Bl, Cl;
        long double Ar, Br, Cr;
        Point2d ndl, nul, nur, ndr=dr;
        double back_ndrx=dr.x, back_new_drx=0;
        double best_Inaccuracy=1000;
        double best_x=0;
        double min_x=0, max_x=0;
        for (int step=0; step<100; ++step){

            Mat tempStep; image.copyTo(tempStep);
            //1
            double ndl_x=(ndr.x*k+2*c*d.x)/(ndr.x*2*t-k);
            ndl=Point2d(ndl_x, (Cd-Ad*ndl_x)/Bd);
            //2
            ps=shapes;
            Point2f pleft=shapes[1];
            do{
                //cout << "left " << ps.size() << endl;
                Al=ndl.y-pleft.y; Bl=pleft.x-ndl.x; Cl=Al*pleft.x+Bl*pleft.y;
                buffer.clear();
                for (auto p: ps){
                    if (p.x-(Cl-Bl*p.y)/Al<-0.1)
                        buffer.push_back(p);
                }
                ps=buffer;
                if (ps.size()){
                    pleft=ps[0];
                    line(tempStep, ndl, pleft, Scalar(0,0, 255), 2);
                }
            }
            while(ps.size());
            line(tempStep, ndl, pleft, Scalar(0,0, 255), 2);
            Al=ndl.y-pleft.y; Bl=pleft.x-ndl.x; Cl=Al*pleft.x+Bl*pleft.y;
            cout << "last left point: " << distance(shapes.begin(), find(shapes.begin(), shapes.end(), pleft)) << endl;
            //3
            pV=Point2f(d.x, (Cl-Al*d.x)/Bl);
            //4
            nul=PointAcross2Lines(Au, Bu, Cu, Al, Bl, Cl);
            //5
            double nur_x=(nul.x*k+2*c*d.x)/(nul.x*2*t-k);
            nur=Point2d(nur_x, (Cu-Au*nur_x)/Bu);
            ps=shapes;
            int steps=0;
            Point2f pright=shapes[15];
            do{
                //cout << "right " << ps.size() << endl;
                Ar=nur.y-pright.y; Br=pright.x-nur.x; Cr=Ar*pright.x+Br*pright.y;
                buffer.clear();
                for (auto p: ps){
                    if ((Cr-Br*p.y)/Ar-p.x<-0.1){
                        buffer.push_back(p);
                    }
                }
                if (buffer.size()==ps.size()){
                    cout << "strange\n";
                    for (int j=0; j<buffer.size(); ++j)
                        cout << ps[j] << " " << buffer[j] <<" : " << distance(shapes.begin(), find(shapes.begin(), shapes.end(), ps[j]))<< endl;
                    throw "ERR";
                }
                ps=buffer;
                if (ps.size()){
                    pright=ps[rand()%ps.size()];
                    ++steps;
                }
            }
            while(ps.size());
            Ar=nur.y-pright.y; Br=pright.x-nur.x; Cr=Ar*pright.x+Br*pright.y;
            Point2d new_ndr=PointAcross2Lines(Ar, Br, Cr, Ad, Bd, Cd);
            double Inaccuracy=ndr.x-new_ndr.x;
            cout << Inaccuracy << " " << new_ndr.x << " " << ndr.x << endl;
            //ndr=Point2f(ndr.x-1, (Cd-Ad*(ndr.x-1))/Bd);
            /*
            if ((max(back_new_drx, back_ndrx)-new_ndr.x<-0.1 && back_ndrx!=ndr.x && back_new_drx!=0) || (new_ndr.x-min(back_new_drx, back_ndrx)<-0.1 && back_ndrx!=ndr.x && back_new_drx!=0)){
                cout << "back_new_drx: (" << back_new_drx << "), back_ndrx: (" << back_ndrx << "), new_ndr.x: (" << new_ndr.x << ")" << endl;
                cout << "start iterate from " << min(back_new_drx, back_ndrx) << " to " << max(back_new_drx, back_ndrx) << endl;
                for (double x=min(back_new_drx, back_ndrx); x<max(back_new_drx, back_ndrx); x+=maxInaccuracy/2){
                    //1
                    double ndl_x=(x*k+2*c*d.x)/(x*2*t-k);
                    ndl=Point2d(ndl_x, (Cd-Ad*ndl_x)/Bd);
                    //2
                    ps=shapes;
                    Point2f pleft=shapes[1];
                    do{
                        //cout << "left " << ps.size() << endl;
                        Al=ndl.y-pleft.y; Bl=pleft.x-ndl.x; Cl=Al*pleft.x+Bl*pleft.y;
                        buffer.clear();
                        for (auto p: ps){
                            if (p.x-(Cl-Bl*p.y)/Al<-0.1)
                                buffer.push_back(p);
                        }
                        ps=buffer;
                        if (ps.size())
                            pleft=ps[0];
                    }
                    while(ps.size());
                    Al=pV.y-pleft.y; Bl=pleft.x-pV.x; Cl=Al*pleft.x+Bl*pleft.y;
                    //3
                    pV=Point2f(d.x, (Cl-Al*d.x)/Bl);
                    //4
                    nul=PointAcross2Lines(Au, Bu, Cu, Al, Bl, Cl);
                    //5
                    double nur_x=(nul.x*k+2*c*d.x)/(nul.x*2*t-k);
                    nur=Point2d(nur_x, (Cu-Au*nur_x)/Bu);
                    ps=shapes;
                    int steps=0;
                    Point2f pright=pV;
                    do{
                        //cout << "right " << ps.size() << endl;
                        Ar=nur.y-pright.y; Br=pright.x-nur.x; Cr=Ar*pright.x+Br*pright.y;
                        buffer.clear();
                        for (auto p: ps){
                            if ((Cr-Br*p.y)/Ar-p.x<-0.1){
                                buffer.push_back(p);
                            }
                        }
                        if (buffer.size()==ps.size()){
                            cout << "strange\n";
                            for (int j=0; j<buffer.size(); ++j)
                                cout << ps[j] << " " << buffer[j] <<" : " << distance(shapes.begin(), find(shapes.begin(), shapes.end(), ps[j]))<< endl;
                            throw "ERR";
                        }
                        ps=buffer;
                        if (ps.size()){
                            pright=ps[rand()%ps.size()];
                            ++steps;
                        }
                    }
                    while(ps.size());
                    Ar=nur.y-pright.y; Br=pright.x-nur.x; Cr=Ar*pright.x+Br*pright.y;
                    Point2d new_ndr=PointAcross2Lines(Ar, Br, Cr, Ad, Bd, Cd);
                    double Inaccuracy=x-new_ndr.x;
                    if (abs(Inaccuracy)<abs(best_Inaccuracy)){
                        best_Inaccuracy=Inaccuracy;
                        best_x=x;
                    }
                }
                break;
            }
            /*
            if ((new_ndr.x>max_x && max_x!=
        line(image, dl, dr, Scalar(255, 0,0), 2);
        line(image, dr, ur, Scalar(255, 0,0), 2);
        line(image, ul, ur, Scalar(255, 0,0), 2);
        line(image, ul, dl, Scalar(255, 0,0), 2);0) || (new_ndr.x<min_x && min_x!=0)){
                cout << "cuted!!!!!" << endl;
                //double ndr_x=min_x+((min_x-max_x)*(min_x*(Au*Bd-Ad*Bu)+Cd*Bu-Cu*Bd))/((min_x+max_x)*(Au*Bd-Ad*Bu)+2*Cd*Bu-2*Cu*Bd);
                double ndr_x=min_x;
                back_x=ndr.x;
                ndr=Point2d(ndr_x, (Cd-Ad*ndr_x)/Bd);
                continue;
            }
            if (sign(ndr.x-back_x)+sign(new_ndr.x-ndr.x) == 0){

                cout << "set limit ";
                back_x=new_ndr.x;
                if (sign(ndr.x-back_x)==-1){cout << "min" << endl; min_x=ndr.x;}
                else if(sign(ndr.x-back_x)==1){cout << "max" << endl;  max_x=ndr.x;}
            }
            back_x=ndr.x;
            */
            /*
            if (abs(Inaccuracy)<abs(best_In
        line(image, dl, dr, Scalar(255, 0,0), 2);
        line(image, dr, ur, Scalar(255, 0,0), 2);
        line(image, ul, ur, Scalar(255, 0,0), 2);
        line(image, ul, dl, Scalar(255, 0,0), 2);accuracy)){
                best_Inaccuracy=Inaccuracy;
                best_x=ndr.x;
            }
            back_ndrx=ndr.x;
            back_new_drx=new_ndr.x;
            */
        /*

            line(tempStep, ndl, ndr, Scalar(0, 255, 0), 2);
            line(tempStep, ndl, nul, Scalar(0, 255, 0), 2);
            line(tempStep, nul, nur, Scalar(0, 255, 0), 2);
            line(tempStep, nur, new_ndr, Scalar(0, 255, 0), 2);
            line(tempStep, new_ndr, ndr, Scalar(255,255,255), 2);
            imshow("step", tempStep);
            waitKey();
            ndr=new_ndr;
            //ndr=(new_ndr+ndr)/2;




            if (abs(Inaccuracy)<maxInaccuracy) break;
        }
        cout << best_Inaccuracy << " " << best_x << endl;
        for (double x=d.x; x<ndr.x*2-d.x; x+=maxInaccuracy/2){
            //1
            double ndl_x=(x*k+2*c*d.x)/(x*2*t-k);
            ndl=Point2d(ndl_x, (Cd-Ad*ndl_x)/Bd);
            //2
            ps=shapes;
            Point2f pleft=shapes[1];
            do{
                //cout << "left " << ps.size() << endl;
                Al=ndl.y-pleft.y; Bl=pleft.x-ndl.x; Cl=Al*pleft.x+Bl*pleft.y;
                buffer.clear();
                for (auto p: ps){
                    if (p.x-(Cl-Bl*p.y)/Al<-0.1)
                        buffer.push_back(p);
                }
                ps=buffer;
                if (ps.size())
                    pleft=ps[0];
            }
            while(ps.size());
            Al=pV.y-pleft.y; Bl=pleft.x-pV.x; Cl=Al*pleft.x+Bl*pleft.y;
            //3
        line(image, ur, um, Scalar(0,255,0), 2);
            pV=Point2f(d.x, (Cl-Al*d.x)/Bl);
            //4
            nul=PointAcross2Lines(Au, Bu, Cu, Al, Bl, Cl);
            //5
            double nur_x=(nul.x*k+2*c*d.x)/(nul.x*2*t-k);
            nur=Point2d(nur_x, (Cu-Au*nur_x)/Bu);
            ps=shapes;
            int steps=0;
            Point2f pright=pV;
            do{
                //cout << "right " << ps.size() << endl;
                Ar=nur.y-pright.y; Br=pright.x-nur.x; Cr=Ar*pright.x+Br*pright.y;
                buffer.clear();
                for (auto p: ps){
                    if ((Cr-Br*p.y)/Ar-p.x<-0.1){
                        buffer.push_back(p);
                    }
                }
                if (buffer.size()==ps.size()){
                    cout << "strange\n";
                    for (int j=0; j<buffer.size(); ++j)
                        cout << ps[j] << " " << buffer[j] <<" : " << distance(shapes.begin(), find(shapes.begin(), shapes.end(), ps[j]))<< endl;
                    throw "ERR";
                }
                ps=buffer;
                if (ps.size()){
                    pright=ps[rand()%ps.size()];
                    ++steps;
                }
            }
            while(ps.size());
            Ar=nur.y-pright.y; Br=pright.x-nur.x; Cr=Ar*pright.x+Br*pright.y;
            Point2d new_ndr=PointAcross2Lines(Ar, Br, Cr, Ad, Bd, Cd);
            double Inaccuracy=x-new_ndr.x;
            if (abs(Inaccuracy)<abs(best_Inaccuracy)){
                best_Inaccuracy=Inaccuracy;
                best_x=x;
            }
        }
        cout << best_Inaccuracy << " " << best_x << " from brutforce" << endl;

    /*
        double some_x=(Cu-Bu*up)/Au;
        Point2d some_up(some_x, up);
        some_x=(Cd-Bd*down)/Ad;
        Point2d some_down(some_x, down);
        double x_mid=right+(left-right)/(kHorizontal+1);
        pV=PointAcross2Lines(some_up, some_down, Point2d(x_mid, (Cu-Bu*x_mid)/Bu), Point2d(x_mid, (Cd-Bd*x_mid)/Bd));
        line(image, some_up, some_down, Scalar(0,0,0), 2);
        ps=shapes;
        long double Al, Bl, Cl;
        long double Ar, Br, Cr;
        if (pV.y<pup.y || pV.y> pdown.y){
            Point2f pleft=shapes[1];
            do{
                cout << "left " << ps.size() << endl;
                Al=pV.y-pleft.y; Bl=pleft.x-pV.x; Cl=Al*pleft.x+Bl*pleft.y;
                buffer.clear();
                for (auto p: ps){
                    if ((Cl-Bl*p.y)/Al>p.x)
                        buffer.push_back(p);
                }
                ps=buffer;
                if (ps.size())
                    pleft=ps[0];
            }
            while(ps.size());
            Al=pV.y-pleft.y; Bl=pleft.x-pV.x; Cl=Al*pleft.x+Bl*pleft.y;

            ps=shapes;
            Point2f pright=shapes[15];
            do{
                cout << "right " << ps.size() << endl;
                Ar=pV.y-pright.y; Br=pright.x-pV.x; Cr=Ar*pright.x+Br*pright.y;
                buffer.clear();
                for (auto p: ps){
                    if ((Cr-Br*p.y)/Ar<p.x)
                        buffer.push_back(p);
                }
                ps=buffer;
                if (ps.size())
                    pright=ps[0];
            }
            while(ps.size());
            Ar=pV.y-pright.y; Br=pright.x-pV.x; Cr=Ar*pright.x+Br*pright.y;

            Point2d vur=PointAcross2Lines(Au, Bu, Cu, Ar, Br, Cr);
            Point2d vul=PointAcross2Lines(Au, Bu, Cu, Al, Bl, Cl);
            Point2d vdr=PointAcross2Lines(Ad, Bd, Cd, Ar, Br, Cr);
            Point2d vdl=PointAcross2Lines(Ad, Bd, Cd, Al, Bl, Cl);

            line(image, vdl, vdr, Scalar(0,255,0), 2);
            line(image, vdr, vur, Scalar(0,255,0), 2);
            line(image, vul, vur, Scalar(0,255,0), 2);
            line(image, vul, vdl, Scalar(0,255,0), 2);
        }
    */
    /*
        long double ndl_x, ndr_x;
        //Point2f pl=shapes[1], pr=shapes[15];
        Point2f pl=ul, pr=ur;
        //ul=Point2d(0, 0.5); ur=Point2d(3.5, 0); dl=Point2d(0, 5); dr=Point2d(3.5, 4);
        //d=Point2d(2.3, 4.3);p22, p2m
        long double Au=ur.y-ul.y, Bu=ul.x-ur.x, Cu=Au*ul.x+Bu*ul.y;
        long double Ad=dr.y-dl.y, Bd=dl.x-dr.x, Cd=Ad*dl.x+Bd*dl.y;
        //cout << "(Au: " << Au << ") (Bu: "  << Bu << ") (Cu: " << Cu << ")\n(Ad: " << Ad << ") (Bd: " << Bd << ") (Cd: " << Cd << ")" << endl;
        long double A=Ad, B=Bd, C=Cd;

        long double t=Au*Bd-Ad*Bu, c=Cd*Bu-Cu*Bd;
        long double k=t*d.x-c;
        long double dtdl=(C-A*pr.x)/B-pr.y, dtdr=(C-A*pl.x)/B-pl.y, dtc=C*(pr.x-pl.x)/B+pr.y*pl.x-pl.y*pr.x;
        long double dlk=pr.x*dtdr, drk=pl.x*dtdl, dlrk=(-A/B)*(pl.x-pr.x)+(pr.y-pl.y);
        cout << "(dtdl: " << dtdl << ") (dtdr: "  << dtdl << ") (dtc: " << dtc << ")\n(dlk: " << dlk << ") (drk: " << drk << ") (dlrk: " << dlrk << ")" << endl;
        //double kl=(dlk-dtdl*d.x), kr=(drk-dtdl*d.x), klr=(dlrk-dtdlr*d.x), kc=(dkc-dtc*d.x);
        long double coef_a=(-dlrk*k-2*t*(drk-dtdr*d.x)); //-(k*klr+2*t*kr);
        long double coef_b=k*(dlk+drk-d.x*(dtdl+dtdr))-dlrk*2*d.x*c-2*t*dtc*d.x;
        long double coef_c=d.x*(2*c*(dlk-dtdl*d.x)+k*dtc);

        cout << setprecision(6);
        cout << "coefs 1 way: (" << float(coef_a) << ") (" << float(coef_b) << ") (" << float(coef_c) << ")\nПроверка 1=" << -d.x*(coef_b+coef_a*d.x)/coef_c << endl;
        //cout << "Корень при прохождении проверки: " << -coef_b/coef_a-d.x << ",  " << (Cd/Bd-Ad/Bd*(-coef_b/coef_a-d.x)) << endl;
        long double D=(coef_b*coef_b-4*coef_a*coef_c);
        cout << "Дискриминант " << D << " " << ((D>=0)?"Positve": "Negative") << endl;
        //if (D<0) throw runtime_error("Нет корней");
        Point2d ndl, ndr;
        if (D>0){// throw runtime_error("2 корня для точки, как узнать какая нужна");
            Point2f ndr1((-coef_b+sqrt(D))/(2*coef_a), (C-A*(-coef_b+sqrt(D))/(2*coef_a))/B);
            Point2f ndr2((-coef_b-sqrt(D))/(2*coef_a), (C-A*(-coef_b-sqrt(D))/(2*coef_a))/B);
            ndr=(std::abs(ndr1.x-d.x)<0.01 ? ndr2 : ndr1);
            cout << "Возможная кординаты 1 способ new_x_r: (" << ndr1.x << ", " << ndr1.y << "), (" << ndr2.x << ", " << ndr2.y << ")" << endl;
        }
        long double Ar=ndr.x*(-A/B)+(C/B-pr.y), Br=pr.x-ndr.x, Cr=ndr.x*(-A/B*pr.x-pr.y)+C/B*pr.x;
        long double pV_y=(Cr-Ar*d.x)/Br;
        cout << "D:(" << d.x << ", " << d.y << ")\npV_y1=" << pV_y << "\n";
        ndl_x=(ndr.x*k+2*d.x*c)/(ndr.x*2*t-k);
        ndl=Point2d(ndl_x, (C-A*ndl_x)/B);
        cout << "ndr:(" << ndr.x << ", " << ndr.y << ")\ndl:(" << ndl.x << ", " << ndl.y << ")" <<endl;
        line(image, pl, ndl, Scalar(255,255,255), 2);
        line(image, pr, ndr, Scalar(255,255,255), 2);
        //cout << (ndl.x*(Au*Bd-Ad*Bu)/Bu/Bd+(Cd*Bu-Cu*Bd)/Bd/Bu)/(ndr.x*(Au*Bd-Ad*Bu)/Bu/Bd+(Cd*Bu-Cu*Bd)/Bd/Bu) << " = " << (d.x-ndl.x)/(ndr.x-d.x) << "\n\n";
        /*

        dtdl=(Cu-Au*pr.x)/Bu-pr.y; dtdr=(Cu-Au*pl.x)/Bu-pl.y; dtc=Cu*(pr.x-pl.x)/Bu+pr.y*pl.x-pl.y*pr.x;
        dlk=pr.x*dtdr; drk=pl.x*dtdl; dlrk=(-Au/Bu)*(pl.x-pr.x)+(pr.y-pl.y);
        cout << "(dtdl: " << dtdl << ") (dtdr: "  << dtdl << ") (dtc: " << dtc << ")\n(dlk: " << dlk << ") (drk: " << drk << ") (dlrk: " << dlrk << ")" << endl;
        //double kl=(dlk-dtdl*d.x), kr=(drk-dtdl*d.x), klr=(dlrk-dtdlr*d.x), kc=(dkc-dtc*d.x);
        coef_a=(-dlrk*k-2*t*(drk-dtdr*d.x)); //-(k*klr+2*t*kr);
        coef_b=k*(dlk+drk-d.x*(dtdl+dtdr))-dlrk*2*d.x*c-2*t*dtc*d.x;
        coef_c=d.x*(2*c*(dlk-dtdl*d.x)+k*dtc);

        cout << "coefs up: (" << float(coef_a) << ") (" << float(coef_b) << ") (" << float(coef_b) << ")\nПроверка 1=" << -d.x*(coef_b+coef_a*d.x)/coef_c << endl;
        //cout << "Корень при прохождении проверки: " << -coef_b/coef_a-d.x << ",  " << (Cd/Bd-Ad/Bd*(-coef_b/coef_a-d.x)) << endl;
        D=(coef_b*coef_b-4*coef_a*coef_c);
        cout << "Дискриминант " << D << " " << ((D>=0)?"Positve": "Negative") << endl;
        //if (D<0) throw runtime_error("Нет корней");
        Point2d nul, nur;
        if (D>0){// throw runtime_error("2 корня для точки, как узнать какая нужна");
            Point2f nur1((-coef_b+sqrt(D))/(2*coef_a), (Cd-Ad*(-coef_b+sqrt(D))/(2*coef_a))/Bd);
            Point2f nur2((-coef_b-sqrt(D))/(2*coef_a), (Cd-Ad*(-coef_b-sqrt(D))/(2*coef_a))/Bd);
            nur=(std::abs(nur1.x-d.x)<1 ? nur2 : nur1);
            cout << "Возможная кординаты 1 способ new_x_r: (" << nur1.x << ", " << nur1.y << "), (" << nur2.x << ", " << nur2.y << ")" << endl;
        }
        Ar=nur.x*(-Au/Bu)+(Cu/Bu-pr.y), Br=pr.x-nur.x, Cr=nur.x*(-Au/Bu*pr.x-pr.y)+Cu/Bu*pr.x;
        pV_y=(Cr-Ar*d.x)/Br;
        cout << "D:(" << d.x << ", " << d.y << ")\npV_y1=" << pV_y << "\n";
        double nul_x=(nur.x*k+2*d.x*c)/(nur.x*2*t-k);
        nul=Point2d(nul_x, (Cu-Au*nul_x)/Bu);
        line(image, pl, nul, Scalar(255,255,255), 2);
        line(image, pr, nur, Scalar(255,255,255), 2);
        */

    /*
        long double alk=(-Ad/Bd), alkc=(Cd/Bd-pl.y), clk=(-Ad*pl.x/Bd-pl.y), clkc=(Cd*pl.x/Bd);
        long double ark=(-Ad/Bd), arkc=(Cd/Bd-pr.y), crk=(-Ad*pr.x/Bd-pr.y), crkc=(Cd*pr.x/Bd);
        dtdl=(pr.x*alk+arkc); dtdr=(pl.x*ark+alkc); double dtdlr=0; dtc=(alkc*pr.x-arkc*pl.x);
        dlk=(clk*pr.x+crkc); drk=(crk*pl.x+clkc); dlrk=(clk-crk); double dkc=0;
        //cout << "(dtdl: " << dtdl << ") (dtdr: "  << dtdl << ") (dtc: " << dtc << ")\n(dlk: " << dlk << ") (drk: " << drk << ") (dlrk: " << dlrk << ")" << endl;
        long double kl=(dlk-dtdl*d.x), kr=(drk-dtdl*d.x), klr=(dlrk-dtdlr*d.x), kc=(dkc-dtc*d.x);
        coef_a=-(k*klr+2*t*kr);
        coef_b=k*(kl+kr)+2*(t*kc-d.x*c*klr);
        coef_c=2*d.x*c*kl-k*kc;

        cout << "coefs 2 way: (" << coef_a << ") (" << coef_b << ") (" << coef_c << ")\nПроверка 1=" << -d.x*(coef_b+coef_a*d.x)/coef_c << endl;
        D=(coef_b*coef_b-4*coef_a*coef_c);
        //cout << "Дискриминант " << D << " " << ((D>=0)?"Positve": "Negative") << endl;
        //if (D<0) throw runtime_error("Нет корней");
        if (D>0){// throw runtime_error("2 корня для точки, как узнать какая нужна");
            Point2f ndr1((-coef_b+sqrt(D))/(2*coef_a), (Cd-Ad*(-coef_b+sqrt(D))/(2*coef_a))/Bd);
            Point2f ndr2((-coef_b-sqrt(D))/(2*coef_a), (Cd-Ad*(-coef_b-sqrt(D))/(2*coef_a))/Bd);
            ndr=(ndr1.x<d.x ? ndr2 : ndr1);
            cout << "Возможная кординаты 2 способ new_x_r: (" << ndr1.x << ", " << ndr1.y << "), (" << ndr2.x << ", " << ndr2.y << ")" << endl;
        }
        Ar=ndr.x*(-Ad/Bd)+(Cd/Bd-pr.y); Br=pr.x-ndr.x; Cr=ndr.x*(-Ad/Bd*pr.x-pr.y)+Cd/Bd*pr.x;
        pV_y=(Cr-Ar*d.x)/Br;long double
        //cout << "D:(" << d.x << ", " << d.y << ")\npV_y2=" << pV_y << "\n";
        ndl_x=(ndr.x*k+2*d.x*c)/(ndr.x*2*t-k);
        ndl=Point2f(ndl_x, (Cd-Ad*ndl_x)/Bd);
        cout << "ndr:(" << ndr.x << ", " << ndr.y << ")\nndl:(" << ndl.x << ", " << ndl.y << ")" <<endl;
        line(image, pl, ndl, Scalar(255,255,255), 2);
        line(image, pr, ndr, Scalar(255,255,255), 2);
    */
        //Point2d pV(d.x, float(pV_y));
        //Point2d nul=PointAcross2Lines(pV, ndl, ul, ur);
        //Point2d nur=PointAcross2Lines(pV, ndr, ul, ur);




        //ndr_x=((ndl_x)(d.x*v-t)+2*d.x+t)/(2*ndl_x*v-d.x*v+t);
        /*
        Point2d downL, downR, leftV, rightV, mid, sunlight, vect, add(50, 50);
        Mat test1=Mat::zeros(612, 612, CV_8UC3), test2=Mat::zeros(612, 612, CV_8UC3);
        vector<Point2f> test1Ps(shapes.size());
        vector<Point2f> test2Ps(shapes.size());
        for (int i=0; i<shapes.size(); ++i){
            test1Ps[i]=Point2f(add)+512*unperspectivePoint(FR.euqlideDistance(dl,ul)/FR.euqlideDistance(dr,ur), dl, dr, ul, ur, shapes[i]);
            test2Ps[i]=Point2f(add)+512*unperspectivePoint((ur.x-d.x)/(d.x-ul.x), dl, dr, ul, ur, shapes[i]);
        }
        drawFacemarks(test1, test1Ps);
        drawFacemarks(test2, test2Ps);
        rectangle(test1, Rect(add, Size(512, 512)), Scalar(0,0,0));
        rectangle(test2, Rect(add, Size(512, 512)), Scalar(0,0,0));
        imshow("k=hl/hr", test1);
        imshow("k=(trueMid-left)/(right-true_mid)", test2);
        */
    /*

        downL=Point2d(FromPerspective(kHorizontal, left, right, PointAcross2Lines(shapes[1], shapes[4], Point2f(dl), Point2f(dr)).x), 1.0);
        downR=Point2d(FromPerspective(kHorizontal, left, right, PointAcross2Lines(shapes[15], shapes[14], Point2f(dl), Point2f(dr)).x), 1.0);
        leftV=Point2d(0, (shapes[1].y-ul.y)/(dl.y-ul.y));
        rightV=Point2d(1, (shapes[15].y-ur.y)/(dr.y-ur.y));

        mid=leftV+(rightV-leftV)/2;
        Point2f vect1(downL-leftV), vect2(downR-rightV);

        sunlight=Point2f(PointAcross2Lines(Point2f(mid), Point2f(mid)+vect1+vect2, Point2f(0,1), Point2f(1,1)));


        line(test, leftV*512+add, downL*512+add, Scalar(0, 255, 0), 2);
        line(test, rightV*512+add, downR*512+add, Scalar(0, 0, 255), 2);
        line(test, Point2d(256,0)+add, Point2d(256,512)+add, Scalar(255, 255, 255));
        line(test, mid*512+add, sunlight*512+add, Scalar(255,0,0), 2);
        double MidX=ToPerspective(kHorizontal, left, right, sunlight.x);
        Point2f perspectiveSunlight1=dr-(dr-dl)*(right-MidX)/(right-left);
        Point2f mid1=shapes[15]+(shapes[1]-shapes[15])/(kHorizontal+1);
        //cout << "(left/right): " << left << "/" << right << endl;


        downL=Point2d(FromPerspective(kHorizontal, left, right, d.x), 1.0);
        downR=downL;
        Point2d p1=shapes[5], p2=shapes[11];

        //cout << "(p1/p2): " << p1 << "/" << p2 << endl;
        Point2d p1l=PointAcross2Lines(p1, pH, Point2d(left, 0), Point2d(left, 5));
        Point2d p2r=PointAcross2Lines(pH, p2, Point2d(right, 0), Point2d(right, 5));

        //cout << "(p1l/p2r): " << p1l << "/" << p2r << endl;
        double localx1=FromPerspective(kHorizontal, left, right, left+(right-left)*(p1.x-p1l.x)/(p2r.x-p1l.x)),
               localx2=FromPerspective(kHorizontal, left, right, left+(right-left)*(p2.x-p1l.x)/(p2r.x-p1l.x));
        leftV=Point2d(localx1,
                      (p1l.y-ul.y)/(dl.y-ul.y));
        rightV=Point2d(localx2,
                      (p2r.y-ur.y)/(dr.y-ur.y));
        //cout << "ul.y: " << ul.y << endl;
        //cout << "leftV: " << leftV << endl;
        //cout << "rightV: " << rightV << endl;
        mid=leftV+(rightV-leftV)/2;
        vect=(downL-leftV)+(downR-rightV);
        //cout << "vect: " << vect << endl << endl;
        sunlight=PointAcross2Lines(mid, mid+vect, Point2d(0,1), Point2d(1,1));

        line(test, leftV*512+add, downL*512+add, Scalar(0, 255, 0), 2);
        line(test, rightV*512+add, downR*512+add, Scalar(0, 0, 255), 2);
        line(test, Point2d(256,0)+add, Point2d(256,512)+add, Scalar(255, 255, 255));
        line(test, mid*512+add, sunlight*512+add, Scalar(255,0,0), 2);
        imshow("test", test);


        MidX=ToPerspective(kHorizontal, left, right, sunlight.x);

        Point2f perspectiveSunlight2=dr-(dr-dl)*(right-MidX)/(right-left);
        Point2f mid2=p1l+(p2r-p1l)*ToPerspective(kHorizontal, left, right, (localx2+localx1)/2.0)/(right-left);

        line(image, mid1, perspectiveSunlight1, Scalar(0, 0,0), 2);
        //line(image, mid2, perspectiveSunlight2, Scalar(0, 0,0), 2);

        Point2f pV=PointAcross2Lines(mid1, perspectiveSunlight1, shapes[8], shapes[27]);
        //pV=PointAcross2Lines(mid1, perspectiveSunlight1, mid2, perspectiveSunlight2);
        Point2f nul, nur, ndl, ndr;
        nul=PointAcross2Lines(pV, shapes[left_ind], Point2f(ul), Point2f(ur));
        nur=PointAcross2Lines(pV, shapes[right_ind], Point2f(ul), Point2f(ur));
        ndl=PointAcross2Lines(pV, shapes[left_ind], Point2f(dl), Point2f(dr));
        ndr=PointAcross2Lines(pV, shapes[right_ind], Point2f(dl), Point2f(dr));

        line(image, shapes[27], pV, Scalar(255, 255,0), 1);
        line(image, nur, pV, Scalar(255, 255,0), 1);
        line(image, nul, pV, Scalar(255, 255,0), 1);
        //line(image, nul, ndl, Scalar(0, 0,0), 2);

        double newK=FR.euqlideDistance(ul, dl)/FR.euqlideDistance(ur, dr);
        Point2f um2=ur+(ul-ur)/(newK+1), dm2=dr+(dl-dr)/(newK+1);

        cout << "newK: " << newK << endl;
        line(image, um2, dm2, Scalar(255,255,255), 1);
    */
    /*
        line(image, dl, dr, Scalar(255, 0,0), 2);
        line(image, dr, ur, Scalar(255, 0,0), 2);
        line(image, ul, ur, Scalar(255, 0,0), 2);
        line(image, ul, dl, Scalar(255, 0,0), 2);

        //line(image, mid1, perspectiveSunlight1, Scalar(0, 0,0), 2);
        //

        //Point2f um=ur+(ul-ur)/(kHorizontal+1), dm=dr+(dl-dr)/(kHorizontal+1);
    /*
        cout << "kHorizontal: " << kHorizontal << endl;
        line(image, um, dm, Scalar(0,0,0), 2);
        const double kVertical=1;//FR.euqlideDistance(ul, dl)/FR.euqlideDistance(ur, dr);
    */
        imshow("Perspective Scene", image);
        //waitKey();
        /* 1: 2
         * 2: 4
         * 3: 8
         * 4: 16
         * 5: 32
         * 6: 64s
         * 7: 128
         * 8: 256
         * 9: 512
        */
        //Mat im=mySuperAlgo(vector<Point2f>({ul, ur, dl, dr}), kHorizontal, kVertical, input, 256, 5);
        //Mat im2=mySuperAlgo(vector<Point2f>({ul, ur, dl, dr}), kHorizontal, kVertical, input, 256, 0);
        Mat im3=myUnperspectiveAlgo(vector<Point2f>({ul, ur, um, dl, dr, dm}),
                                    vector<double>{ Aul, Bul, Cul,
                                                    Adl, Bdl, Cdl,
                                                    Aur, Bur, Cur,
                                                    Adr, Bdr, Cdr},
                                    pHl, pHr, input, 512, 2);
        Mat im=myUnperspectiveAlgo(vector<Point2f>({ul, ur, um, dl, dr, dm}),
                                    vector<double>{ Aul, Bul, Cul,
                                                    Adl, Bdl, Cdl,
                                                    Aur, Bur, Cur,
                                                    Adr, Bdr, Cdr},
                                    pHl, pHr, input, 512, 5);
        Mat im2=myUnperspectiveAlgo(vector<Point2f>({ul, ur, um, dl, dr, dm}),
                                    vector<double>{ Aul, Bul, Cul,
                                                    Adl, Bdl, Cdl,
                                                    Aur, Bur, Cur,
                                                    Adr, Bdr, Cdr},
                                    pHl, pHr, input, 512, 1);

        //auto matrix=getPerspectiveTransform(p_src, p_dst);
        //cv::warpPerspective(input, im, matrix, Size(512, 512));
        //imshow("Wraped scene 6", im);
        //imshow("Wraped scene 0", im2);
        //imshow("Wraped scene 2", im3);
        //waitKey();
        return im3;
    }


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

    static vector<Rect> GetFaceRects(InputArray& input){
        vector<Rect> faces;
        FaceDetector2(input, faces, nullptr);
        return faces;
    }

    template<class T>
    static double euqlideDistance(T f, T s){
        return sqrt(pow<double>(f.x-s.x, 2)+pow<double>(f.y-s.y, 2));
    }

    static pair<Mat, Rect> TakeFaceMask(InputArray& input){
        Mat image_inp; input.copyTo(image_inp);
        vector<vector<Point2f> > shapes;
        if(!facemark->fit(image_inp, vector<Rect>{Rect(0, 0, image_inp.cols, image_inp.rows)}, shapes)){
            return {Mat::zeros(1,1, CV_8UC1), Rect()};
        }

        Mat output = Mat::zeros(Size(image_inp.cols, image_inp.rows), CV_8UC1);
        output+=100;
        const float FloodFillTolerance = 0.06;
        Rect rect;

        for (int c=1; c<=16; ++c){
            line(output, shapes[0][c-1], shapes[0][c], Scalar(255,0,0), 3);
        }
        line(output, shapes[0][26], shapes[0][16], Scalar(255,0,0), 3);
        line(output, shapes[0][26], shapes[0][25], Scalar(255,0,0), 3);
        line(output, shapes[0][24], shapes[0][25], Scalar(255,0,0), 3);

        line(output, shapes[0][24], shapes[0][19], Scalar(255,0,0), 3);

        line(output, shapes[0][18], shapes[0][19], Scalar(255,0,0), 3);
        line(output, shapes[0][18], shapes[0][17], Scalar(255,0,0), 3);
        line(output, shapes[0][17], shapes[0][0], Scalar(255,0,0), 3);

        vector<Point2f> seedPoints={
            Point2f(0, 0),
            Point2f(output.cols-1, 0),
            (shapes[0][4]-shapes[0][33])*1.3+shapes[0][33],
            (shapes[0][12]-shapes[0][33])*1.3+shapes[0][33],
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
        threshold(
            output,
            output,
            0,
            255,
            THRESH_TOZERO
        );
        output.convertTo(output, CV_8UC1, 255);
        int min_x=output.cols, max_x=0, min_y=output.rows, max_y=0;
        for (auto p: shapes[0]){
            int x=p.x;
            int y=p.y;
            if (min_x>x) min_x=x;
            if (min_y>y) min_y=y;
            if (max_x<x) max_x=x;
            if (max_y<y) max_y=y;
        }
        max_x=min(output.cols, max_x);
        max_y=min(output.rows, max_y);
        min_x=max(0, min_x);
        min_y=max(8, min_y);
        return {output, Rect(Point(min_x, min_y), Point(max_x, max_y))};

        //cvtColor(image_inp, image_inp, COLOR_BGR2GRAY);

    }

    static double calculateMeanBrightness(const cv::Mat& image) {
        cv::Scalar mean = cv::mean(image);
        return mean[0];  // Для одноканального изображения, или mean.val[0] для цветного
    }

    static vector<pair<Mat, Rect>> NormalizeRotate(InputArray& input, InputArray& faces){
        Mat image_inp; input.copyTo(image_inp);
        vector<pair<Mat, Rect>> answer;
        vector<Rect> rects; faces.copyTo(rects);
        future<tuple<bool, Mat, Rect>> single_responce[rects.size()];
        for (int i=0; i<rects.size(); ++i){
            Rect rec=rects[i];
            single_responce[i]=async(launch::async, [image_inp](Rect rec) -> tuple<bool, Mat, Rect>{

                vector<vector<Point2f> > shapes;
                if(!facemark->fit(image_inp, vector<Rect>{rec}, shapes)){
                    return {false, Mat::zeros(1,1, CV_8UC1), Rect()};
                }
                Mat image; image_inp.copyTo(image);
                int x1,x2,y1,y2;
                x1=rec.x; x2=rec.x+rec.width;
                y1=rec.y; y2=rec.y+rec.height;
                Point2f p1=shapes[0][27], p2=shapes[0][8];
                auto deltaX=p2.x-p1.x, deltaY=p2.y-p1.y;
                Point centre((p2-p1)/2+p1);
                double angle=-atan(deltaX/deltaY);
                auto M = getRotationMatrix2D(centre, (angle)*180.0/3.14, 1.0);
                double total=angle;

                Point p[4]={

                    Point(rec.x+rec.width, rec.y+rec.height),
                    Point(rec.x, rec.y+rec.height),
                        Point(rec.x, rec.y),
                        Point(rec.x+rec.width, rec.y)};
                double len[4];
                double angle_p[4];
                for (int i = 0; i < 4; ++i) {
                    len[i]=euqlideDistance(centre, p[i]);
                }
                angle_p[0]=acos((p[0].x-centre.x)/len[0])-total;
                angle_p[1]=acos((p[1].x-centre.x)/len[1])-total;
                angle_p[2]=3.1415+acos((centre.x-p[2].x)/len[2])-total;
                angle_p[3]=3.1415*2-acos((p[3].x-centre.x)/len[3])-total;


                double max_x=0, min_x=image_inp.cols, max_y=0, min_y=image_inp.rows;
                for (int i = 0; i < 4; ++i) {
                    double x=len[i]*cos(angle_p[i])+centre.x;
                    double y=len[i]*sin(angle_p[i])+centre.y;
                    if (max_x<x) max_x=x;
                    if (min_x>x) min_x=x;
                    if (max_y<y) max_y=y;
                    if (min_y>y) min_y=y;
                }
                max_x=std::min<int>(max_x, image_inp.cols);
                max_y=std::min<int>(max_y, image_inp.rows);
                min_x=std::max<int>(min_x, 0);
                min_y=std::max<int>(min_y, 0);

                //19-24 line
                Rect face(Point(min_x, min_y), Point(max_x, max_y));
                warpAffine(image, image, M, Size(image.cols, image.rows));

                return {true, image, face};
            }, rects[i]);
        }

        for (int i=0; i<rects.size(); ++i){
            auto status = single_responce[i].get();
            if (get<0>(status)){
                answer.push_back({get<1>(status), get<2>(status)});
            }
        }

        return answer;
    }



    inline static cv::Ptr<cv::face::FacemarkLBF> facemark = cv::face::FacemarkLBF::create();

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



    static bool KeyPointCompare(KeyPoint p1, KeyPoint p2){
        return p1.response > p2.response;
    }




    inline static cv::dnn::Net net = cv::dnn::readNetFromCaffe("/home/nikita/Загрузки/deploy.prototxt", "/home/nikita/Загрузки/res10_300x300_ssd_iter_140000_fp16.caffemodel");
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
                    if (Grayscale){
                        images.push_back(imread(path, IMREAD_GRAYSCALE));
                    }
                    else
                        images.push_back(imread(path));
                    //
                    labels.push_back(atoi(classlabel.c_str()));
                }catch(...){
                }
            }

        }
    }

    void func_get_image(string path){
        ifstream st(path);
        ofstream o("LBPH_people.csv");
        string temp;
        while(!st.eof()){
            getline(st, temp);
            if (temp=="") break;
            o << temp << " " << temp.substr(string_view("/home/nikita/QtProj/GetPictureStudents/Faces/Face_").length(), temp.length()-4) << "\n";
        }
        o.close();
    }

    static bool FaceDetector2(InputArray& input, OutputArray& f, void *conf, double ctotal=0){
        double total=ctotal;
        double marign=0.10;
        Mat image;
        input.copyTo(image);
        resize(image, image, Size(300,300), 0,0, INTER_CUBIC);
        //imshow("I", image);
        Mat inputBlob = cv::dnn::blobFromImage(image, 1);

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
               faces.push_back(Rect(Point(x1,y1), Point(x2,y2)));
                //if (faces.empty() || (faces[0].width<x2-x1 && faces[0].height<y2-y1))
               //     faces = {Rect(Point(x1,y1), Point(x2,y2))};

            }
        }
        double angle=10;
        //total=0.0;
        if (faces.empty() || faces.size()==0){
            auto M = getRotationMatrix2D(Point(input.cols()/2,input.rows()/2), angle, 1.0);
            input.copyTo(image);
            warpAffine(image, image, M, Size(image.cols, image.rows));
            total+=angle;
            if (total>=120) return false;
            return FaceDetector2(image, f, conf, total);
        }



        input.copyTo(image);
        vector<Rect> ans;
        for (int i=0; i<faces.size(); i++){
        faces[i].x*=input.cols()/300.0; faces[i].width*=input.cols()/300.0;
        faces[i].y*=input.rows()/300.0; faces[i].height*=input.rows()/300.0;


        x1=faces[i].x; x2=faces[i].x+faces[i].width;
        y1=faces[i].y; y2=faces[i].y+faces[i].height;
        Point centre(x1+(x2-x1)/2,y1+(y2-y1)/2);
        auto l=(sqrt(pow(x2-x1,2)+pow(y2-y1,2)))/2;
        auto a=acos((x1-centre.x)/l);
        auto x1t=l*cos(total+a), y1t=l*sin(total+a);
        a=acos((x2-centre.x)/l);
        auto x2t=l*cos(total+a), y2t=l*sin(total+a);
        auto x=max(abs(x1t), abs(x2t)), y=max(abs(y1t), abs(y2t));
        Rect rect(Point(max<double>(0, centre.x-x), max<double>(0,centre.y-y)),
                  Point(min<double>(image.cols, centre.x+x), min<double>(image.rows,centre.y+y)));
           ans.push_back(rect);
        }
        Mat(ans).copyTo(f);
        return true;
    }

   static bool defaultFaceDetector(InputArray& input, OutputArray& f, void *conf){
       return FaceDetector2(input, f, conf);
   }
};

#endif // MYFACERECOGNITION_H
