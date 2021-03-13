#include "Resnet50TF.h"

Resnet50TF::Resnet50TF(TFConfig tfconfig) : TFModel(tfconfig)
{
	key = "Resnet50TF_" + key;
}

//对分数进行下手
std::vector<ResnetDST> Resnet50TF::OUT2DST(std::vector<tensorflow::Tensor>& out)
{
	std::vector<ResnetDST> ret;
	auto scoreTensor = out[0].tensor<float, 2>();
	for (int i = 0; i < out[0].dim_size(0); i++)
	{
		float score = (scoreTensor(i, 0));
		ret.emplace_back(score);
	}
	return ret;
}