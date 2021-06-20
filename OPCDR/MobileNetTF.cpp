#include "MobileNetTF.h"

MobileNetTF::MobileNetTF(TFConfig tfconfig) : TFModel(tfconfig)
{
	key = "MobileNetTF_" + key;
}

std::vector<MobileNetDST> MobileNetTF::OUT2DST(std::vector<tensorflow::Tensor>& out)
{
	std::vector<MobileNetDST> ret;
	auto scoreTensor = out[0].tensor<float, 2>();
	for (int i = 0; i < out[0].dim_size(0); i++)
	{
		float score = (scoreTensor(i, 0));
		ret.emplace_back(score);
	}
	return ret;
}
