#pragma once
#include "TRModel.h"

#include "Model1.h"

class TRModel1 : public TRModel<M1DST>
{
public:
	TRModel1(TRConfig trconfig);
	virtual std::vector<M1DST> OUT2DST(TROUT& out);
	//���ǵ�ԭʼ�Ĵ������磬Ϊ����Ӧ��ʵ�����Խ���model1
	virtual void constructNetwork();
};

