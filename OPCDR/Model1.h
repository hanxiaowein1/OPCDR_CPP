#pragma once

#ifndef _OPCDR_MODEL1_H_
#define _OPCDR_MODEL1_H_
#include <vector>
#include "opencv2/opencv.hpp"

class model1Result
{
public:
	float score;//model1的分数
	std::vector<cv::Point> points;//定位点

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