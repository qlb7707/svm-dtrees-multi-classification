/*================================================================
*   Copyright (C) 2016 All rights reserved.
*   
*   filename     :train_test.cpp
*   author       :qinlibin
*   create date  :2016/06/29
*   mail         :qin_libin@foxmail.com
*
================================================================*/
//#include "train_test.h"
#include <opencv2/opencv.hpp>
#include "common.h"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#define FEAT_SMALL_IMAGE
#ifdef FEAT_SMALL_IMAGE
#define FEAT_SIZE 8*8
#endif

using namespace std;
using namespace cv;

Mat trainData;
Mat trainLabel;

int str2int(string src)
{
    int num;
    istringstream iss(src);
    iss>>num;
    return num;
}

string int2str(int n)
{
    ostringstream oss;
    string str;
    oss<<n;
    str = oss.str();
    return str;
}
void train_mnist_svm()
{
    int n_img = count_dir_files("train_image");
    if(n_img <= 0)
    {
        cout << "No image avaliable"<<endl;
        return;
    }
    ifstream ifs;
    ifs.open("train_label.txt");
    if(!ifs)
    {
        cout<<"train_label.txt missing"<<endl;
        return;
    }
    string prefix = "train_image/train_";
    Mat tmp = Mat(8,8,CV_8UC1);
    Mat img_feat = Mat(1,FEAT_SIZE,CV_32FC1);
    trainData = Mat::zeros(n_img,FEAT_SIZE,CV_32FC1);
    trainLabel = Mat::zeros(n_img,1,CV_32SC1);
    for(int i = 0; i < n_img; i++)
    {
//        cout<<"iter:"<< i <<endl;
        string line,name;
        int label;
        if(!getline(ifs,line))
        {
            cout<<"image number and labels do not match"<<endl;
            return;
        }
        istringstream iss(line);
        iss >> name >> label;
        string fname = prefix + int2str(i)+".jpg";
        Mat img = imread(fname.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
        resize(img,tmp,tmp.size());
        for(int j = 0; j < 8; j++)
        {
            for(int k = 0; k < 8; k++)
            {
                img_feat.data[j * 8 + k] = tmp.at<uchar>(j,k);
            }
        }
        normalize(img_feat,img_feat);
        memcpy(trainData.data + i * FEAT_SIZE * sizeof(float),img_feat.data,FEAT_SIZE * sizeof(float));
        trainLabel.at<unsigned int>(i,0) = label;
    }

    //start_training
    //
    cout<<"start training"<<endl;
    CvSVM svm;
    CvSVMParams params;
    CvTermCriteria criteria;

    criteria = cvTermCriteria(CV_TERMCRIT_EPS,1000,0.01);//FLT_EPSILON);
    params = CvSVMParams(CvSVM::C_SVC,CvSVM::RBF,10.0,8.0,1.0,10.0,0.5,0.1,NULL,criteria);
    svm.train(trainData,trainLabel,Mat(),Mat(),params);
    cout<<"Done !"<<endl;
    svm.save("svm_train.xml");
    ifs.close();
    return;

}

int main()
{
     train_mnist_svm();
   // cout<<count_dir_files("test_image")<<endl;
    return 0;
}
