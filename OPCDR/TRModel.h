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

/// <summary>
/// TensorRTģ������
/// </summary>
class TRConfig {
public:
	unsigned long long memory;
	std::string modelPath;
	std::string quantize;
	std::string enginePath;
	std::string quantizeEnginePath;
	int batchsize;
	int channel;
	int height;
	int width;
	std::string inputName;
	std::vector<std::string> outputName;
	std::vector<int> outputSize;//����output�Ĵ�С

	std::string modelname;
public:
	TRConfig() {};
	TRConfig(const char* iniPath)
	{
		height = IniConfig::instance().getIniInt("ModelInput", "height");
		width = IniConfig::instance().getIniInt("ModelInput", "width");
		channel = IniConfig::instance().getIniInt("ModelInput", "channel");
		batchsize = IniConfig::instance().getIniInt("ModelInput", "batchsize");
		modelPath = IniConfig::instance().getIniString("TR", "path");
		initializeEnginePath();
		//����modelPath�Ʋ��enginePath
		inputName = IniConfig::instance().getIniString("TR", "input");
		std::string temp = IniConfig::instance().getIniString("TR", "output");
		outputName = split(temp, ',');
		memory = IniConfig::instance().getIniInt("TR", "memory");
		std::string temp2 = IniConfig::instance().getIniString("TR", "output_size");
		auto temp3 = split(temp2, ',');
		for (auto elem : temp3) {
			outputSize.emplace_back(std::stoi(elem));
		}

		quantize = IniConfig::instance().getIniString("TR", "quantize");
	}

	TRConfig(const char* iniPath, std::string name)
	{
		modelname = name;
		quantize = IniConfig::instance().getIniString("TensorRT", "quantize");
		if (modelname == "resnet50") {
			height = IniConfig::instance().getIniInt("Resnet50", "height");
			width = IniConfig::instance().getIniInt("Resnet50", "width");
			channel = IniConfig::instance().getIniInt("Resnet50", "channel");
			batchsize = IniConfig::instance().getIniInt("Resnet50", "batchsize");

			modelPath = IniConfig::instance().getIniString("Resnet50TR", "path");
			initializeEnginePath();

			inputName = IniConfig::instance().getIniString("Resnet50TR", "input");
			std::string temp = IniConfig::instance().getIniString("Resnet50TR", "output");
			outputName = split(temp, ',');
			memory = IniConfig::instance().getIniInt("Resnet50TR", "memory");
			std::string temp2 = IniConfig::instance().getIniString("Resnet50TR", "output_size");
			auto temp3 = split(temp2, ',');
			for (auto elem : temp3) {
				outputSize.emplace_back(std::stoi(elem));
			}
		}
		if (modelname == "mobilenet") {
			height = IniConfig::instance().getIniInt("MobileNet", "height");
			width = IniConfig::instance().getIniInt("MobileNet", "width");
			channel = IniConfig::instance().getIniInt("MobileNet", "channel");
			batchsize = IniConfig::instance().getIniInt("MobileNet", "batchsize");

			modelPath = IniConfig::instance().getIniString("MobileNetTR", "path");
			initializeEnginePath();

			inputName = IniConfig::instance().getIniString("MobileNetTR", "input");
			std::string temp = IniConfig::instance().getIniString("MobileNetTR", "output");
			outputName = split(temp, ',');
			memory = IniConfig::instance().getIniInt("MobileNetTR", "memory");
			std::string temp2 = IniConfig::instance().getIniString("MobileNetTR", "output_size");
			auto temp3 = split(temp2, ',');
			for (auto elem : temp3) {
				outputSize.emplace_back(std::stoi(elem));
			}
		}
	}

	//TensorRT���Ա���engine�ļ���engine�ļ�Ϊ��gpu�г�ʼ�����ģ�ͣ�����ʡȥ��ʼ��ģ�͵�ʱ�䣬�˴�Ϊ��ʼ��engine·��
	void initializeEnginePath() {
		//enginePath = getFileNamePrefix(&modelPath);
		std::string parentPath = getFileParentPath(modelPath);
		std::string prefix = getFileNamePrefix(modelPath);
		enginePath = parentPath + "\\" + prefix + ".engine";
		quantizeEnginePath = parentPath + "\\" + prefix + "_16.engine";
	}

};

//int�����м���ͼ��vector������ͼ�������
using TRIN = std::pair<int, std::vector<float>>;
//int�����м���ͼ��vector��������vector�е�vector������ͼ���һ����������ϣ������������д��һ��vector�е�
using TROUT = std::pair<int, std::vector<std::vector<float>>>;

/// <summary>
/// �̳���Model���TensorRT��ʵ����SRC2IN()��run()����
/// </summary>
/// <typeparam name="DST"></typeparam>
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
	//TensorRT���ͨ����Tensorflowͨ���в��죬��Ҫ��ͼ�����NCHW->NHWC�����෴��
	bool transformInMemory(vector<cv::Mat>& imgs, float* dstPtr);
	void build();
	virtual void constructNetwork();
public:
	virtual TRIN SRC2IN(std::vector<cv::Mat>& src);
	virtual TROUT run(TRIN& in);

private:
	//����engine�ļ���ʼ�����磬���engine�ļ������ڣ�����ѡ�񱣴�engine�ļ����ڳ�ʼ��
	void buildEngine();
	//����engine�ļ�·����ʼ������
	void buildEngine(std::string enginePath);
	//�������絽engine�ļ�
	void saveEngine(std::string enginePath);
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
void TRModel<DST>::buildEngine()
{
	if (mTrConfig.quantize == "ON") {
		if (checkFileExists(mTrConfig.quantizeEnginePath)) {
			buildEngine(mTrConfig.quantizeEnginePath);
		}
		else {
			mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mBuilder->buildCudaEngine(*mNetwork), samplesCommon::InferDeleter());
			//����engine�ļ�
			saveEngine(mTrConfig.quantizeEnginePath);
		}
	}
	else {
		if (checkFileExists(mTrConfig.enginePath)) {
			buildEngine(mTrConfig.enginePath);
		}
		else {
			mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mBuilder->buildCudaEngine(*mNetwork), samplesCommon::InferDeleter());
			//����engine�ļ�
			saveEngine(mTrConfig.enginePath);
		}
	}
}

template<typename DST>
void TRModel<DST>::saveEngine(std::string enginePath)
{
	if (mEngine) 
	{
		IHostMemory* serialized_model = mEngine->serialize();
		std::ofstream serialize_output_stream(enginePath.c_str(), std::fstream::out | std::fstream::binary);
		if (serialize_output_stream) {
			serialize_output_stream.write((char*)serialized_model->data(), serialized_model->size());
		}
		serialize_output_stream.close();
		serialized_model->destroy();
	}
}

template<typename DST>
void TRModel<DST>::buildEngine(std::string enginePath)
{
	std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
	std::streamsize size = file.tellg();
	//std::unique_ptr<char[]> buffer(new char[size]);
	char* buffer = new char[size];
	file.seekg(0, std::ios::beg);
	if (!file.read(buffer, size)) {
		std::cout << "read file to buffer failed" << endl;
	}
	IRuntime* runtime = createInferRuntime(gLogger);
	//mEngine.reset(runtime->deserializeCudaEngine(buffer.get(), size, nullptr));
	//mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mBuilder->buildCudaEngine(*mNetwork), samplesCommon::InferDeleter());
	mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(buffer, size, nullptr), samplesCommon::InferDeleter());
	delete[]buffer;
}

template<typename DST>
void TRModel<DST>::build()
{
	mBuilder.reset(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));//����Ҫ����ͬʱ��һ��ȫ�ֱ�����ʼ�����ģ�ͻ᲻�����
	mNetwork.reset(mBuilder->createNetwork());
	mConfig.reset(mBuilder->createBuilderConfig());
	mParser.reset(nvuffparser::createUffParser());

	mTrConfig.memory = mTrConfig.memory * (1 << 30);
	mBuilder->setMaxWorkspaceSize(mTrConfig.memory);

	constructNetwork();

	mBuilder->setMaxBatchSize(mTrConfig.batchsize);
	if (mTrConfig.quantize == "ON") {
		mBuilder->setFp16Mode(true);
	}
	buildEngine();
	//mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mBuilder->buildCudaEngine(*mNetwork), samplesCommon::InferDeleter());
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

	//ע��˳����CHW������HWC
	for (int i = 0; i < imgs.size(); i++) {
		for (int c = 0; c < channel; c++) {
			for (int h = 0; h < height; h++) {
				float* linePtr = (float*)imgs[i].ptr(h);
				for (int w = 0; w < width; w++) {
					//�����ַ
					dstPtr[i * height * width * channel + c * height * width + h * width + w] = *(linePtr + w * 3 + c);
				}
			}
		}
	}
	return true;
}

#endif