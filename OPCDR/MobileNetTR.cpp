#include "MobileNetTR.h"

MobileNetTR::MobileNetTR(TRConfig trconfig) :TRModel(trconfig)
{
	key = "MobileNetTR_" + key;
	TRModel<MobileNetDST>::build();
}

std::vector<MobileNetDST> MobileNetTR::OUT2DST(TROUT& out)
{
	std::vector<MobileNetDST> ret;
	for (int i = 0; i < out.first; i++)
	{
		MobileNetDST result;
		result = out.second[0][i];
		ret.emplace_back(result);
	}
	return ret;
}