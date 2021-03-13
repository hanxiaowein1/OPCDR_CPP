#pragma once
#ifndef _OPCDR_TFMODEL1_H_
#define _OPCDR_TFMODEL1_H_
#include "TFModel.h"
#include "Model1.h"
//typename std::vector<model1Result> M1DST;

class TFModel1 : public TFModel<M1DST>
{
private:
	std::vector<model1Result> resultOutput(std::vector<tensorflow::Tensor>& tensors);
	void TensorToMat(tensorflow::Tensor mask, cv::Mat* dst);
	std::vector<cv::Point> getRegionPoints2(cv::Mat& mask, float threshold);
public:
	TFModel1(TFConfig tfconfig);
	virtual std::vector<M1DST> OUT2DST(std::vector<tensorflow::Tensor>& out);
};

#endif