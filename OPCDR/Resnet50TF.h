#pragma once

#ifndef _OPCDR_RESNET50TF_H_
#define _OPCDR_RESNET50TF_H_
#include "TaskThread.h"
#include "TFModel.h"
#include "Resnet50.h"

//��׼��resnet50��tensorflowʵ��(��ʱʹ��model2���в��ԣ���Ϊmodel2ֻ��һ�����������Ե��������resnet50)
class Resnet50TF : public TFModel<ResnetDST>
{
private:
public:
	Resnet50TF(TFConfig tfconfig);
public:
	//ʵ��OUT2DST
	virtual std::vector<ResnetDST> OUT2DST(std::vector<tensorflow::Tensor>& out);

public:

};

#endif