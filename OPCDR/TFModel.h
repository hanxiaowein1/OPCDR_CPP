#pragma once
#ifndef _OPCDR_TFMODEL_H_
#define _OPCDR_TFMODEL_H_

#include<iostream>
#include<string>
#include<vector>
#include <chrono>
#include <memory>
#include "Eigen/Dense"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>

#include "opencv2/opencv.hpp"
#include "Model.h"
#include "IniConfig.h"
#include "commonFunction.h"

class TFConfig 
{
public:
	std::string modelPath;
	std::string inputName;
	std::vector<std::string> outputName;
	int height;
	int width;
	int channel;
public:
	TFConfig() {}
	TFConfig(const char* ini_path)
	{
		height = IniConfig::instance().getIniInt("ModelInput", "height");
		width = IniConfig::instance().getIniInt("ModelInput", "width");
		channel = IniConfig::instance().getIniInt("ModelInput", "channel");
		modelPath = IniConfig::instance().getIniString("ModelProperty", "path");
		inputName = IniConfig::instance().getIniString("ModelProperty", "input");
		std::string temp = IniConfig::instance().getIniString("ModelProperty", "output");
		outputName = split(temp, ',');
	}

};

//typename tensorflow::Tensor tt;

template<typename DST>
class TFModel : public Model<tensorflow::Tensor, std::vector<tensorflow::Tensor>, cv::Mat, DST>
{
public:
	std::unique_ptr<tensorflow::Session> m_session;
	TFConfig mTfConfig;
public:
	TFModel(TFConfig tfconfig);
public:
	virtual tensorflow::Tensor SRC2IN(std::vector<cv::Mat>& src);
	virtual std::vector<tensorflow::Tensor> run(tensorflow::Tensor& in);
	//virtual std::vector<DST> run(std::vector<cv::Mat>& imgs) = 0;
	virtual std::vector<DST> OUT2DST(std::vector<tensorflow::Tensor>& out) = 0;

	virtual ~TFModel();
public:
	void Mat2Tensor(std::vector<cv::Mat>& imgs, tensorflow::Tensor& tensor);
};

template<typename DST>
TFModel<DST>::TFModel(TFConfig tfconfig):mTfConfig(tfconfig)
{
	//mTfConfig = tfconfig;
	tensorflow::GraphDef graph_def;
	tensorflow::Status load_graph_status =
		ReadBinaryProto(tensorflow::Env::Default(),
			mTfConfig.modelPath,
			&graph_def);
	if (!load_graph_status.ok()) {
		std::cout << mTfConfig.modelPath << ": [LoadGraph] load graph failed!\n";
		return;
	}

	tensorflow::SessionOptions options;
	options.config.mutable_device_count()->insert({ "GPU",1 });
	options.config.mutable_gpu_options()->set_allow_growth(true);
	options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
	m_session.reset(tensorflow::NewSession(options));
	auto status_creat_session = m_session.get()->Create(graph_def);
	std::cout << mTfConfig.modelPath << "create session success\n";
	if (!status_creat_session.ok()) {
		std::cout << "[LoadGraph] creat session failed!\n" << std::endl;
		return;
	}
}

template<typename DST>
TFModel<DST>::~TFModel()
{
	m_session->Close();
	
	//tensorflow::SessionOptions options;
	//options.config.mutable_device_count()->insert({ "GPU",1 });
	//options.config.mutable_gpu_options()->set_allow_growth(false);
	//std::vector<string> containers;
	//tensorflow::Reset(options, containers);
}

template<typename DST>
void TFModel<DST>::Mat2Tensor(std::vector<cv::Mat>& imgs, tensorflow::Tensor& tensor)
{
	int size = imgs.size();
	if (size == 0)
		return;
	int height = imgs[0].rows;
	int width = imgs[0].cols;
	int channel = imgs[0].channels();
	for (int i = 0; i < size; i++)
	{
		float* ptr = tensor.flat<float>().data() + i * height * width * channel;
		cv::Mat tensor_image(height, width, CV_32FC3, ptr);
		imgs[i].convertTo(tensor_image, CV_32F);//转为float类型的数组
		tensor_image = (tensor_image / 255 - 0.5) * 2;
	}
}

template<typename DST>
tensorflow::Tensor TFModel<DST>::SRC2IN(std::vector<cv::Mat>& src)
{	
	int size = src.size();
	if (size == 0)
	{
		std::cout << "has zero" << std::endl;
	}
	int height = src[0].rows;
	int width = src[0].cols;
	int channel = src[0].channels();
	tensorflow::Tensor in(tensorflow::DataType::DT_FLOAT,
		tensorflow::TensorShape({ size, height, width, channel }));
	Mat2Tensor(src, in);
	return in;
}

template<typename DST>
std::vector<tensorflow::Tensor> TFModel<DST>::run(tensorflow::Tensor& in)
{
	std::vector<tensorflow::Tensor> out;
	auto status_run = m_session->Run({ { mTfConfig.inputName, in } },
		mTfConfig.outputName, {}, &out);
	if (!status_run.ok()) {
		std::cout << "run model failed!\n";
	}
	return out;
}

#endif