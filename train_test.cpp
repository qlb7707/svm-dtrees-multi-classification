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
#include "flag_def.h" 
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

void train_by_svm(int debug)
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
    /*parameters of SVM
     * int svm_type             --CvSVM::C_SVC,CvSVM::NU_SVC,CvSVM::ONE_CLASS,CvSVM::EPS_SVR,CvSVM::NU_SVR
     * int kernel_type          --CvSVM::LINEAR,CvSVM::POLY,CvSVM::RBF,CvSVM::SIGMOID
     * double degree            --degree of poly kernel
     * double gamma             --gamma of poly/rbf/sgmoid kernel
     * double coef0             --coef0 of poly/sigmoid kernel
     * double Cvalue            --C of problem C_SVC/EPS_SVR/NU_SVR
     * double nu                --nu of problem NU_SVC/ONE_CLASS/NU_SVR
     * double p                 --p of problem EPS_SVR
     * CvMat* class_weights     --optional weights in C_SVC problem,assigned to particular classes.
     * CvTermCriteria term_crit --termination criteria
     * */
    params = CvSVMParams(CvSVM::C_SVC,CvSVM::RBF,10.0,8.0,1.0,10.0,0.5,0.1,NULL,criteria);
    svm.train(trainData,trainLabel,Mat(),Mat(),params);
    cout<<"Done !"<<endl;
    svm.save(SVM_MODEL_NAME);
    trainData.release();
    trainLabel.release();
    return;

}

void test_by_svm(int debug)
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
    cout<<"Done !"<<endl;
    ifs.close();
    ofs.close();
}

void train_by_rtrees(int debug)
{
    if(prepare_data(debug))
    {
         cout<<"prepare data failed"<<endl;
         return;
    }
    cout<<"start training Random Trees..."<<endl;
    CvRTrees forest;
    CvRTParams params;
    /*random forest parameters
     *int max_depth                       -- the depth of the tree
     *int min_sample_count                -- minimum samples required at a leaf node for int to split
     *float regression_accuracy
     *bool use_surrogate
     *int max_categories
     *const float* priors
     *bool calc_var_importance            -- if this is true then variable importance will be calculated
     *int nactive_vars                    -- the size of the randomly selected subset of features at each tree node used to find best splits
     *int max_num_of_trees_in_the_forest  -- the maximum number of trees in the forest
     *float forest_accuracy               -- sufficient accuracy(oob error)
     *int termcrit_type                   -- terminal type     */
    params = CvRTParams(10,10,0,false,15,0,true,10,500,0.01f,CV_TERMCRIT_ITER);
    forest.train(trainData,CV_ROW_SAMPLE,trainLabel,Mat(),Mat(),Mat(),Mat(),params);
    cout<<"Done !"<<endl;
    forest.save(RANDOM_TREES_MODEL_NAME);
    trainData.release();
    trainLabel.release();
    return;
}

void test_by_rtrees(int debug)
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
    CvRTrees forest;
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
        res = (int)forest.predict(img_feat);
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
    cout<<"Done !"<<endl;
    ifs.close();
    ofs.close();
}
int main(int argc,char *argv[])
{
    void (*train)(int);
    void (*test)(int);
    train = train_by_svm;
    test = test_by_svm;
    rtc::FlagList::SetFlagsFromCommandLine(&argc,argv,true);
    if(FLAG_help)
    {
        rtc::FlagList::Print(NULL,false);
    }
    if(string(FLAG_algo) == "svm")
    {
        train = train_by_svm;
        test = test_by_svm;
    }
    else if(string(FLAG_algo) == "rtrees")
    {
        train = train_by_rtrees;
        test = test_by_rtrees;
    }
    else
    {
        cout<<"unsupported algorithm"<<endl;
        return -1;
    }

    if(string(FLAG_mode) == "train")
    {
        train(FLAG_debug);
    }
    else if(string(FLAG_mode) == "test")
    {
        test(FLAG_debug);     
    }
    else
    {
        cout<<"unsupported mode"<<endl;
        return -1;
    }

    return 0;
}
