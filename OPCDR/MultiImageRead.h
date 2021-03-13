#pragma once
#ifndef _OPCDR_MULTIIMAGEREAD_H_
#define _OPCDR_MULTIIMAGEREAD_H_

#include "PopQueueData.h"
#include "SlideRead.h"
#include "SlideFactory.h"
#include "opencv2/opencv.hpp"

using MIRData = std::pair<cv::Rect, cv::Mat>;

class MultiImageRead : public PopQueueData<MIRData>
{
public:
	std::string m_slidePath;
	std::vector<std::unique_ptr<SlideRead>> sReads;
	std::vector<std::mutex> sRead_mutex;
	std::atomic<int> read_level = 0;
	std::atomic<bool> gamma_flag = true;
	int m_threadnum;
public:
	MultiImageRead(const char* slidePath);
	~MultiImageRead();

	void createReadHandle(int num);

	void gammaCorrection(cv::Mat& src, cv::Mat& dst, float fGamma);

	void task(int i, cv::Rect rect);

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