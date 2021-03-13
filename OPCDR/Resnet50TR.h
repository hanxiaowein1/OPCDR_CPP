#pragma once
#ifndef _OPCDR_RESNET50TR_H_
#define _OPCDR_RESNET50TR_H_
#include "TaskThread.h"
#include "TRModel.h"
#include "Resnet50.h"


class Resnet50TR : public TRModel<ResnetDST>
{
public:
	Resnet50TR(TRConfig trconfig);
public:
	virtual std::vector<ResnetDST> OUT2DST(TROUT& out);
};

#endif