#pragma once
#ifndef _OPCDR_MULTIIMAGEREAD_H_
#define _OPCDR_MULTIIMAGEREAD_H_

#include "PopQueueData.h"
#include "SlideRead.h"
#include "SlideFactory.h"
#include "opencv2/opencv.hpp"

using MIRData = std::pair<cv::Rect, cv::Mat>;

/// <summary>
/// 多线程读图类，继承自PopQueueData，可以多线程的存储
/// </summary>
class MultiImageRead : public PopQueueData<MIRData>
{
public:
	std::string m_slidePath;
	//全切片图像的handle
	std::vector<std::unique_ptr<SlideRead>> sReads;
	//每一个handle的锁
	std::vector<std::mutex> sRead_mutex;
	//读取图像的层级
	std::atomic<int> read_level = 0;
	//是否进行gamma变换
	std::atomic<bool> gamma_flag = true;
	int m_threadnum;
public:
	MultiImageRead(const char* slidePath);
	~MultiImageRead();
	//创建读图的handle
	void createReadHandle(int num);
	//gamma变换
	void gammaCorrection(cv::Mat& src, cv::Mat& dst, float fGamma);
	//利用第i个handle，在切片图像中读取rect区域的图像
	void task(int i, cv::Rect rect);
	//新增读图任务
	void read(std::vector<cv::Rect> rects);

public:
	//level0下的宽
	void getSlideWidth(int& width);
	//level0下的高
	void getSlideHeight(int& height);
	//level0下的有效区域x轴起始点
	void getSlideBoundX(int& boundX);
	//level0下的有效区域y轴起始点
	void getSlideBoundY(int& boundY);
	//获得mpp
	void getSlideMpp(double& mpp);
	//获取指定level下的宽，高
	void getLevelDimensions(int level, int& width, int& height);
	//为了方便起见，在MultiImageRead里面也设置单张图像读取
	void getTile(int level, int x, int y, int width, int height, cv::Mat& img);

	//获得各个图层的比例
	int get_ratio();

	void setReadLevel(int level) {
		read_level = level;
	}
};

#endif