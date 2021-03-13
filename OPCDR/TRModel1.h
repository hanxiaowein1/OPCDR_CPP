#pragma once
#include "TRModel.h"

#include "Model1.h"

class TRModel1 : public TRModel<M1DST>
{
public:
	TRModel1(TRConfig trconfig);
	virtual std::vector<M1DST> OUT2DST(TROUT& out);
	//覆盖掉原始的创建网络，为了适应本实验室自建的model1
	virtual void constructNetwork();
};

