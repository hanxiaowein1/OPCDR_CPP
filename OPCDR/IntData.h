#pragma once
#include "PopQueueData.h"
class IntData : public PopQueueData<int>
{
public:
	void task();
	void process();
};

