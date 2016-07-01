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
#include <unistd.h>
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
void train_mnist_svm(int debug)
{
    int n_img = count_dir_files("train_image");
    if(n_img <= 0)
    {
        cout << "No image avaliable"<<endl;
        return;
    }
    ifstream ifs;
    ofstream ofs;
    ifs.open("train_label.txt");
    if(debug)
    {
        ofs.open("train_debug.txt");
    }
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
                img_feat.at<float>(0,j * 8 + k) = tmp.at<uchar>(j,k);
            }
        }
        normalize(img_feat,img_feat);
       if(debug) 
       {
            for(int j = 0; j < 8; j++)
            {
                for(int k = 0; k < 8; k++)
                {
                    ofs << img_feat.at<float>(0,j * 8 + k)<<" ";
                }
            }
            ofs<<endl;
       }
        memcpy(trainData.data + i * FEAT_SIZE * sizeof(float),img_feat.data,FEAT_SIZE * sizeof(float));
        trainLabel.at<unsigned int>(i,0) = label;
    }

    //start_training
    //
    if(debug)
    {
        ofs.close();
    }
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

void test_mnist_svm(int debug)
{
    ifstream ifs;
    ofstream ofs;
    string filename,line;
    Mat img;
    int label,res,cnt = 0,correct = 0;
    ofs.open("predict_result.txt",ios::out);
    cout<<"start testing..."<<endl;
    ifs.open("test_label.txt",ios::in);
    CvSVM svm;
    if(access("svm_train.xml",F_OK))
    {
        cout<<"svm_train.xml missing"<<endl;
        return;
    }
    svm.load("svm_train.xml");
    if(!ifs)
    {
        cout<<"test_label.txt missing"<<endl;
        return;
    }

    Mat tmp = Mat(8,8,CV_8UC1);
    Mat img_feat = Mat(1,FEAT_SIZE,CV_32FC1);
    while(getline(ifs,line))
    {
        istringstream iss(line);
        iss >> filename >> label;
        img = imread(filename.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
        if(!img.data)
            continue;
        resize(img,tmp,tmp.size());
        for(int i = 0; i < 8; i++)
        {
            for(int j = 0; j < 8; j++)
            {
                img_feat.at<float>(0,i * 8 + j) = tmp.at<uchar>(i,j); 
            }
        }
        normalize(img_feat,img_feat);
        if(debug)
        {
            for(int i = 0; i < 8; i++)
            {
                for(int j = 0; j < 8; j++)
                {
                    ofs << img_feat.at<float>(0,i*8 +j)<<" ";
                }
            }
        }
        res = svm.predict(img_feat);
        ofs<< filename <<" "<< res <<" "<< label<<endl;
        if(res == label)
        {
            correct ++;
        }
        cnt ++;
    }
    ofs<<"accurcy:"<<1.0 * correct / cnt << "(" << correct <<"/"<<cnt<<")"<<endl;
    ifs.close();
    ofs.close();
}

int main(int argc,char *argv[])
{
    int train_flag = 0,test_flag = 0, debug_flag = 0;
    string arg1,arg2,arg3;
    if(argc >= 2)
    {
        arg1 = argv[1];
        if(arg1 == "-t")
        {
            train_flag = 1;
        }
        else if(arg1 == "-d")
        {
            debug_flag = 1;
        }
        else if(arg1 == "-p")
        {
            test_flag = 1;
        }
        else
        {
            cout << "param " << arg1 << "not supported" << endl;
            return -1;
        }
    }
    if(argc >= 3)
    {
        arg2 = argv[2];
        if(arg2 == "-t")
        {
            train_flag = 1;
        }
        else if(arg2 == "-d")
        {
            debug_flag = 1;
        }
        else if(arg2 == "-p")
        {
            test_flag = 1;
        }
        else
        {
            cout << "param " << arg1 << "not supported" << endl;
            return -1;
        }
    }
    if(argc == 4)
    {
        arg3 = argv[3];
        if(arg3 == "-t")
        {
            train_flag = 1;
        }
        else if(arg3 == "-d")
        {
            debug_flag = 1;
        }
        else if(arg3 == "-p")
        {
            test_flag = 1;
        }
        else
        {
            cout << "param " << arg1 << "not supported" << endl;
            return -1;
        }
    }


    if(train_flag)
    {
        train_mnist_svm(debug_flag);
    }
    if(test_flag)
    {
        test_mnist_svm(debug_flag);     
    }
   // cout<<count_dir_files("test_image")<<endl;
    return 0;
}
