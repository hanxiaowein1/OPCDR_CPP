#pragma once
#ifndef _OPCDR_TRMODEL_H_
#define _OPCDR_TRMODEL_H_

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>

#include "Model.h"
#include "opencv2/opencv.hpp"
#include "commonFunction.h"
class TRConfig {
public:
	unsigned long long memory;
	std::string modelPath;
	int batchsize;
	int channel;
	int height;
	int width;
	std::string inputName;
	std::vector<std::string> outputName;
	std::vector<int> outputSize;//各个output的大小
public:
	TRConfig() {}
	TRConfig(const char* iniPath)
	{
		height = IniConfig::instance().getIniInt("ModelInput", "height");
		width = IniConfig::instance().getIniInt("ModelInput", "width");
		channel = IniConfig::instance().getIniInt("ModelInput", "channel");
		batchsize = IniConfig::instance().getIniInt("ModelInput", "batchsize");
		modelPath = IniConfig::instance().getIniString("TR", "path");
		inputName = IniConfig::instance().getIniString("TR", "input");
		std::string temp = IniConfig::instance().getIniString("TR", "output");
		outputName = split(temp, ',');
		memory = IniConfig::instance().getIniInt("TR", "memory");
		std::string temp2 = IniConfig::instance().getIniString("TR", "output_size");
		auto temp3 = split(temp2, ',');
		for (auto elem : temp3) {
			outputSize.emplace_back(std::stoi(elem));
		}
	}
};

//int代表有几张图像，vector代表几张图像的数据
using TRIN = std::pair<int, std::vector<float>>;
//int代表有几张图像，vector代表结果，vector中的vector代表几张图像的一个结果的整合，例如分数，是写在一个vector中的
using TROUT = std::pair<int, std::vector<std::vector<float>>>;

template <typename DST>
class TRModel : public Model<TRIN, TROUT, cv::Mat, DST>
{
public:
	template <typename T>
	using myUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

	myUniquePtr<nvinfer1::IBuilder> mBuilder{ nullptr };
	myUniquePtr<nvinfer1::INetworkDefinition> mNetwork{ nullptr };
	myUniquePtr<nvinfer1::IBuilderConfig> mConfig{ nullptr };
	myUniquePtr<nvuffparser::IUffParser> mParser{ nullptr };
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine{ nullptr };
	samplesCommon::BufferManager* mBuffer{ nullptr };
	myUniquePtr<nvinfer1::IExecutionContext> mContext{ nullptr };

public:
	TRConfig mTrConfig;
public:
	TRModel(TRConfig);
	~TRModel();
	bool transformInMemory(vector<cv::Mat>& imgs, float* dstPtr);
	void build();
	virtual void constructNetwork();
public:
	virtual TRIN SRC2IN(std::vector<cv::Mat>& src);
	virtual TROUT run(TRIN& in);
};

template<typename DST>
TRModel<DST>::TRModel(TRConfig trconfig)
{
	mTrConfig = trconfig;
}

template<typename DST>
void TRModel<DST>::constructNetwork()
{
	mParser->registerInput(mTrConfig.inputName.c_str(),
		nvinfer1::Dims3(mTrConfig.channel, mTrConfig.height, mTrConfig.width),
		nvuffparser::UffInputOrder::kNCHW);
	for (int i = 0; i < mTrConfig.outputName.size(); i++)
	{
		mParser->registerOutput(mTrConfig.outputName[i].c_str());
	}
	mParser->parse(mTrConfig.modelPath.c_str(), *mNetwork);
}

template<typename DST>
void TRModel<DST>::build()
{
	mBuilder.reset(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));//这里要考虑同时用一个全局变量初始化多个模型会不会出错
	mNetwork.reset(mBuilder->createNetwork());
	mConfig.reset(mBuilder->createBuilderConfig());
	mParser.reset(nvuffparser::createUffParser());

	mTrConfig.memory = mTrConfig.memory * (1 << 30);
	mBuilder->setMaxWorkspaceSize(mTrConfig.memory);

	constructNetwork();

	mBuilder->setMaxBatchSize(mTrConfig.batchsize);
	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mBuilder->buildCudaEngine(*mNetwork), samplesCommon::InferDeleter());
	mContext.reset(mEngine->createExecutionContext());
	mBuffer = new samplesCommon::BufferManager(mEngine, mTrConfig.batchsize);
}

template<typename DST>
inline TRModel<DST>::~TRModel()
{
}

template<typename DST>
TROUT TRModel<DST>::run(TRIN& in)
{
	TROUT ret;
	float* hostInputBuffer = static_cast<float*>((*mBuffer).getHostBuffer(mTrConfig.inputName));
	std::memcpy(hostInputBuffer, in.second.data(), in.second.size() * sizeof(float));
	mBuffer->copyInputToDevice();
	mContext->execute(in.first, mBuffer->getDeviceBindings().data());
	mBuffer->copyOutputToHost();

	ret.first = in.first;

	for (int i = 0; i < mTrConfig.outputName.size(); i++)
	{
		float* output = static_cast<float*>(mBuffer->getHostBuffer(mTrConfig.outputName[i]));
		std::vector<float> sub(ret.first * mTrConfig.outputSize[i]);
		std::memcpy(sub.data(), output, sub.size() * sizeof(float));
		ret.second.emplace_back(sub);
	}
	return ret;
}

template<typename DST>
TRIN TRModel<DST>::SRC2IN(std::vector<cv::Mat>& src)
{
	int size = src.size();
	int height = src[0].rows;
	int width = src[0].cols;
	int channel = src[0].channels();

	vector<float> neededData(height * width * channel * size);
	transformInMemory(src, neededData.data());
	std::pair<int, std::vector<float>> temp_elem;
	temp_elem.first = src.size();
	temp_elem.second = std::move(neededData);
	return temp_elem;
}

template<typename DST>
bool TRModel<DST>::transformInMemory(vector<cv::Mat>& imgs, float* dstPtr)
{
	if (imgs.size() == 0)
		return false;
	int width = imgs[0].cols;
	int height = imgs[0].rows;
	int channel = imgs[0].channels();
	for (int i = 0; i < imgs.size(); i++)
	{
		imgs[i].convertTo(imgs[i], CV_32F);
		imgs[i] = (imgs[i] / 255 - 0.5) * 2;
	}

	//注意顺序，是CHW，不是HWC
	for (int i = 0; i < imgs.size(); i++) {
		for (int c = 0; c < channel; c++) {
			for (int h = 0; h < height; h++) {
				float* linePtr = (float*)imgs[i].ptr(h);
				for (int w = 0; w < width; w++) {
					//换算地址
					dstPtr[i * height * width * channel + c * height * width + h * width + w] = *(linePtr + w * 3 + c);
				}
			}
		}
	}
	return true;
}

#endif