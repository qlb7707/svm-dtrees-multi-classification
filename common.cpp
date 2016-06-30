/*================================================================
*   Copyright (C) 2016 All rights reserved.
*   
*   filename     :commom.cpp
*   author       :qinlibin
*   create date  :2016/06/29
*   mail         :qin_libin@foxmail.com
*
================================================================*/
#include "common.h"
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstring>
#include <iostream>
#include <errno.h>
using namespace std;
int count_dir_files(const char dir[])
{
    DIR *pdir;
    struct dirent *pdirent;
    struct stat f_ftime;
    int fcnt = 0;
    pdir = opendir(dir);
    string prefix = string(dir)+"/";
    if(!pdir)
    {
        cout<<"dir open fail"<<endl;
        return -1;
    }
    for(pdirent = readdir(pdir); pdirent != NULL; pdirent = readdir(pdir))
    {
        if(strcmp(pdirent->d_name,".") == 0 || strcmp(pdirent->d_name,"..") == 0)
        {
            continue;
        }
    //    cout<<pdirent->d_name<<endl;
        if(stat((prefix+pdirent->d_name).c_str(),&f_ftime) != 0)
        {
            cout<<"status error"<<endl;
            if(errno == ENOENT)
            {
                cout<<"file not exist"<<endl;
            }
            if(errno == ENOTDIR)
            {
                cout<<"not true dir"<<endl;
            }
            if(errno == ELOOP)
            {
                cout<<"too many links"<<endl;
            }
            if(errno == EFAULT)
            {
                cout<<"invalid point"<<endl;
            }
            if(errno == ENOMEM)
            {
                cout<<"insufficient memory"<<endl;
            }
            if(errno == ENAMETOOLONG)
            {
                cout<<"name too long"<<endl;
            }
            return -1;
        }
        if(S_ISDIR(f_ftime.st_mode))
        {
            continue;
        }
        fcnt ++;
    }
    closedir(pdir);
    return fcnt;

}
