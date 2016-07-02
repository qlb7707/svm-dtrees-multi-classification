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
#define FEAT_HEIGHT 8
#define FEAT_WIDTH 8
#define FEAT_SIZE FEAT_HEIGHT * FEAT_WIDTH
#endif
#define TRAIN_LABEL_FILENAME "train_label.txt"
#define TEST_LABEL_FILENAME "test_label.txt"
#define TRAIN_IMAGE_DIR "train_image"
#define TEST_IMAGE_DIR "test_image"
#define TRAIN_DEBUG_FILENAME "train_debug.txt"
#define TEST_DEBUG_FILENAME "test_debug.txt"
#define TEST_RESULT_FILENAME "predict_result.txt"
#define SVM_MODEL_NAME "svm_train.xml"
#define RANDOM_TREES_MODEL_NAME "random_tree_train.xml"
using namespace std;
using namespace cv;

Mat trainData;
Mat trainLabel;



void make_feature(Mat src,Mat &dst)
{
    assert(src.data && dst.data && src.type() == CV_8UC1 && 
            dst.type() == CV_32FC1 && dst.size() == Size(FEAT_WIDTH*FEAT_HEIGHT,1));
    Mat tmp = Mat(FEAT_HEIGHT,FEAT_WIDTH,CV_8UC1);
    resize(src,tmp,tmp.size());
    for(int i = 0; i < FEAT_HEIGHT; i++)
    {
        for(int j = 0; j < FEAT_WIDTH; j++)
        {
            dst.at<float>(0,FEAT_WIDTH * i + j) = tmp.at<uchar>(i,j);
        }
    }
    normalize(dst,dst);
}


void write_feature_to_file(Mat feat, ofstream &ofs)
{
    assert(ofs && feat.data && feat.size() == Size(FEAT_WIDTH*FEAT_HEIGHT,1)&& feat.type() == CV_32FC1);
    for(int i = 0; i < FEAT_HEIGHT; i++)
    {
        for(int j = 0; j < FEAT_WIDTH; j++)
        {
            ofs << feat.at<float>(0,i * FEAT_WIDTH +j)<<" ";
        }
    }
    ofs << endl;
}

int prepare_data(int debug)
{
    ifstream ifs;
    ofstream ofs;
    ifs.open(TRAIN_LABEL_FILENAME);
    if(debug)
    {
        ofs.open(TRAIN_DEBUG_FILENAME);
    }
    if(!ifs)
    {
        cout<<TRAIN_LABEL_FILENAME<<" missing"<<endl;
        return -1;
    }
    Mat img_feat = Mat(1,FEAT_SIZE,CV_32FC1);
    string line,name;
    int label;
    while(getline(ifs,line))
    {
        istringstream iss(line);
        iss >> name >> label;
        Mat img = imread(name.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
        if(!img.data)
            continue;
        make_feature(img,img_feat);
        if(debug) 
        {
            write_feature_to_file(img_feat,ofs);
        }
        trainData.push_back(img_feat);
        trainLabel.push_back(label);
    }
    ifs.close();
    ofs.close();
    if(trainData.data && trainLabel.data)
        return 0;
    else
        return -1;
}

void train_mnist_svm(int debug)
{
    if(prepare_data(debug))
    {
        cout<<"prepare data failed"<<endl;
        return;
    }
    //start_training
    //
    cout<<"start training SVM..."<<endl;
    CvSVM svm;
    CvSVMParams params;
    CvTermCriteria criteria;

    criteria = cvTermCriteria(CV_TERMCRIT_EPS,1000,0.001);//FLT_EPSILON);
    params = CvSVMParams(CvSVM::C_SVC,CvSVM::RBF,10.0,8.0,1.0,10.0,0.5,0.1,NULL,criteria);
    svm.train(trainData,trainLabel,Mat(),Mat(),params);
    cout<<"Done !"<<endl;
    svm.save(SVM_MODEL_NAME);
    return;

}

void test_mnist_svm(int debug)
{
    ifstream ifs;
    ofstream ofs,ofs_d;
    string filename,line;
    Mat img;
    int label,res,cnt = 0,correct = 0;
    cout<<"start testing SVM..."<<endl;
    ifs.open(TEST_LABEL_FILENAME,ios::in);
    ofs.open(TEST_RESULT_FILENAME,ios::out);
    if(debug)
    {
        ofs_d.open(TEST_DEBUG_FILENAME);
    }
    CvSVM svm;
    if(access(SVM_MODEL_NAME,F_OK))
    {
        cout<<SVM_MODEL_NAME<<" missing"<<endl;
        return;
    }
    svm.load(SVM_MODEL_NAME);
    if(!ifs)
    {
        cout<<TEST_LABEL_FILENAME<<" missing"<<endl;
        return;
    }

    Mat img_feat = Mat(1,FEAT_SIZE,CV_32FC1);
    while(getline(ifs,line))
    {
        istringstream iss(line);
        iss >> filename >> label;
        img = imread(filename.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
        if(!img.data)
            continue;
        make_feature(img, img_feat);
        if(debug)
        {
            write_feature_to_file(img_feat,ofs_d);
        }
        res = svm.predict(img_feat);
        ofs<< filename <<" "<< res <<" "<< label<<endl;
        if(res == label)
        {
            correct ++;
        }
        cnt ++;
    }
    if(debug)
    {
        ofs_d.close();
    }
    ofs<<"accurcy:"<<1.0 * correct / cnt << "(" << correct <<"/"<<cnt<<")"<<endl;
    ifs.close();
    ofs.close();
}

void train_mnist_dtrees(int debug)
{
    if(prepare_data(debug))
    {
         cout<<"prepare data failed"<<endl;
         return;
    }
    cout<<"start training Random Trees..."<<endl;
    CvDTree forest;
    CvRTParams params;

    params = CvRTParams(10,10,0,false,15,0,true,4,100,0.01f,CV_TERMCRIT_ITER);
    forest.train(trainData,CV_ROW_SAMPLE,trainLabel,Mat(),Mat(),Mat(),Mat(),params);
    cout<<"Done !"<<endl;
    forest.save(RANDOM_TREES_MODEL_NAME);
    return;
}

void test_mnist_dtrees(int debug)
{
    ifstream ifs;
    ofstream ofs,ofs_d;
    string filename,line;
    Mat img;
    int label,res,cnt = 0,correct = 0;
    cout<<"start testing Random Trees..."<<endl;
    ifs.open(TEST_LABEL_FILENAME,ios::in);
    ofs.open(TEST_RESULT_FILENAME,ios::out);
    if(debug)
    {
        ofs_d.open(TEST_DEBUG_FILENAME);
    }
    CvDTree forest;
    if(access(RANDOM_TREES_MODEL_NAME,F_OK))
    {
        cout<<RANDOM_TREES_MODEL_NAME<<" missing"<<endl;
        return;
    }
    forest.load(RANDOM_TREES_MODEL_NAME);
    if(!ifs)
    {
        cout<<TEST_LABEL_FILENAME<<" missing"<<endl;
        return;
    }

    Mat img_feat = Mat(1,FEAT_SIZE,CV_32FC1);
    while(getline(ifs,line))
    {
        istringstream iss(line);
        iss >> filename >> label;
        img = imread(filename.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
        if(!img.data)
            continue;
        make_feature(img, img_feat);
        if(debug)
        {
            write_feature_to_file(img_feat,ofs_d);
        }
        res = forest.predict(img_feat)->value;
        ofs<< filename <<" "<< res <<" "<< label<<endl;
        if(res == label)
        {
            correct ++;
        }
        cnt ++;
    }
    if(debug)
    {
        ofs_d.close();
    }
    ofs<<"accurcy:"<<1.0 * correct / cnt << "(" << correct <<"/"<<cnt<<")"<<endl;
    ifs.close();
    ofs.close();
}
int main(int argc,char *argv[])
{
    int train_flag = 0,test_flag = 0, debug_flag = 0;
    string arg1,arg2,arg3;
    void (*train)(int);
    void (*test)(int);
    train = train_mnist_svm;
    test = test_mnist_svm;
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
        else if(arg1 == "--svm")
        {
            train = train_mnist_svm;
            test = test_mnist_svm;
        }
        else if(arg1 == "--dtrees")
        {
            train = train_mnist_dtrees;
            test = test_mnist_dtrees;
        }
        else
        {
            cout << "param " << arg1 << " not supported" << endl;
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
        else if(arg2 == "--svm")
        {
            train = train_mnist_svm;
            test = test_mnist_svm;
        }
        else if(arg2 == "--dtrees")
        {
            train = train_mnist_dtrees;
            test = test_mnist_dtrees;
        }
        else
        {
            cout << "param " << arg2 << " not supported" << endl;
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
        else if(arg3 == "--svm")
        {
            train = train_mnist_svm;
            test = test_mnist_svm;
        }
        else if(arg3 == "--dtrees")
        {
            train = train_mnist_dtrees;
            test = test_mnist_dtrees;
        }
        else
        {
            cout << "param " << arg3 << " not supported" << endl;
            return -1;
        }
    }


    if(train_flag)
    {
        train(debug_flag);
    }
    if(test_flag)
    {
        test(debug_flag);     
    }
    return 0;
}
