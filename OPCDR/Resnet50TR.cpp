#include "Resnet50TR.h"

Resnet50TR::Resnet50TR(TRConfig trconfig) :TRModel(trconfig)
{
	key = "Resnet50TR_" + key;
	TRModel<ResnetDST>::build();
}


std::vector<ResnetDST> Resnet50TR::OUT2DST(TROUT& out)
{
	std::vector<ResnetDST> ret;
	for (int i = 0; i < out.first; i++)
	{
		ResnetDST result;
		result = out.second[0][i];
		ret.emplace_back(result);
	}
	return ret;
}