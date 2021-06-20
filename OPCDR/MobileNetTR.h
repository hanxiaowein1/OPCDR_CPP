#pragma once
#ifndef _OPCDR_MOBILENETTR_H_
#define _OPCDR_MOBILENETTR_H_

#include "MobileNet.h"
#include "TaskThread.h"
#include "TRModel.h"

class MobileNetTR : public TRModel<MobileNetDST>
{
public:
	MobileNetTR(TRConfig trconfig);
public:
	virtual std::vector<MobileNetDST> OUT2DST(TROUT& out);
};

#endif