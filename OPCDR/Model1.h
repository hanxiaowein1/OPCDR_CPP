#pragma once

#ifndef _OPCDR_MODEL1_H_
#define _OPCDR_MODEL1_H_
#include <vector>
#include "opencv2/opencv.hpp"

class model1Result
{
public:
	float score;//model1�ķ���
	std::vector<cv::Point> points;//��λ��

public:
	bool operator>(const model1Result& result)
	{
		if (score > result.score)
			return true;
		return false;
	}
};



using M1DST = model1Result;

#endif