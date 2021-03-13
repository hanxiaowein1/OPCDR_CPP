#pragma once

#ifndef _OPCDR_RESNET50TF_H_
#define _OPCDR_RESNET50TF_H_
#include "TaskThread.h"
#include "TFModel.h"
#include "Resnet50.h"

//标准的resnet50，tensorflow实现(暂时使用model2进行测试，因为model2只有一个分数，可以当做纯粹的resnet50)
class Resnet50TF : public TFModel<ResnetDST>
{
private:
public:
	Resnet50TF(TFConfig tfconfig);
public:
	//实现OUT2DST
	virtual std::vector<ResnetDST> OUT2DST(std::vector<tensorflow::Tensor>& out);

public:

};

#endif