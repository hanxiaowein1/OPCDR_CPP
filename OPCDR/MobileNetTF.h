#pragma once
#ifndef _OPCDR_MOBILENETTF_H_
#define _OPCDR_MOBILENETTF_H_

#include "TaskThread.h"
#include "TFModel.h"
#include "MobileNet.h"

class MobileNetTF : public TFModel<MobileNetDST>
{
public:
	MobileNetTF(TFConfig tfconfig);
public:
	virtual std::vector<MobileNetDST> OUT2DST(std::vector<tensorflow::Tensor>& out);
};

#endif