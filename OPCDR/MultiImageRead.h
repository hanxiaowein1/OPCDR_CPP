#pragma once
#ifndef _OPCDR_MULTIIMAGEREAD_H_
#define _OPCDR_MULTIIMAGEREAD_H_

#include "PopQueueData.h"
#include "SlideRead.h"
#include "SlideFactory.h"
#include "opencv2/opencv.hpp"

using MIRData = std::pair<cv::Rect, cv::Mat>;

/// <summary>
/// ���̶߳�ͼ�࣬�̳���PopQueueData�����Զ��̵߳Ĵ洢
/// </summary>
class MultiImageRead : public PopQueueData<MIRData>
{
public:
	std::string m_slidePath;
	//ȫ��Ƭͼ���handle
	std::vector<std::unique_ptr<SlideRead>> sReads;
	//ÿһ��handle����
	std::vector<std::mutex> sRead_mutex;
	//��ȡͼ��Ĳ㼶
	std::atomic<int> read_level = 0;
	//�Ƿ����gamma�任
	std::atomic<bool> gamma_flag = true;
	int m_threadnum;
public:
	MultiImageRead(const char* slidePath);
	~MultiImageRead();
	//������ͼ��handle
	void createReadHandle(int num);
	//gamma�任
	void gammaCorrection(cv::Mat& src, cv::Mat& dst, float fGamma);
	//���õ�i��handle������Ƭͼ���ж�ȡrect�����ͼ��
	void task(int i, cv::Rect rect);
	//������ͼ����
	void read(std::vector<cv::Rect> rects);

public:
	//level0�µĿ�
	void getSlideWidth(int& width);
	//level0�µĸ�
	void getSlideHeight(int& height);
	//level0�µ���Ч����x����ʼ��
	void getSlideBoundX(int& boundX);
	//level0�µ���Ч����y����ʼ��
	void getSlideBoundY(int& boundY);
	//���mpp
	void getSlideMpp(double& mpp);
	//��ȡָ��level�µĿ���
	void getLevelDimensions(int level, int& width, int& height);
	//Ϊ�˷����������MultiImageRead����Ҳ���õ���ͼ���ȡ
	void getTile(int level, int x, int y, int width, int height, cv::Mat& img);

	//��ø���ͼ��ı���
	int get_ratio();

	void setReadLevel(int level) {
		read_level = level;
	}
};

#endif