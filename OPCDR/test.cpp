#include "TFModel1.h"
#include "TRModel1.h"
#include "Resnet50TF.h"
#include "Resnet50TR.h"

#include "SlidePredict.h"
#include "IniConfig.h"

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
Model<MLIN, MLOUT, SRC, DST>* get(Model<MLIN, MLOUT, SRC, DST>* model)
{
	return model;
}


void tfTest()
{
	TaskThread task_thread;
	task_thread.createThreadPool(3);


	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
	TFConfig tfconfig;
	tfconfig.modelPath = "D:\\TEST_DATA\\model\\liusibo\\20201127\\model\\L2BD_model1_258_54.pb";
	tfconfig.inputName = "input_1:0";
	tfconfig.outputName.emplace_back("dense_2/Sigmoid:0");
	tfconfig.outputName.emplace_back("conv2d_1/truediv:0");
	tfconfig.height = 512;
	tfconfig.width = 512;
	//TFModel1 *tfmodel1 = new TFModel1(tfconfig);
	//Model<tensorflow::Tensor, std::vector<tensorflow::Tensor>, std::vector<cv::Mat>, std::vector<model1Result>>* model = new TFModel1(tfconfig);

	//auto model = new TFModel1(tfconfig);
	//auto l = get(model);
	auto model = get(new TFModel1(tfconfig));

	cv::Mat img = cv::imread("D:\\TEST_OUTPUT\\rnnPredict\\sfy1148589 0893178\\model2\\0_114139_64427_0.999979.tif");
	cv::resize(img, img, cv::Size(512, 512));

	std::vector<cv::Mat> imgs;
	imgs.emplace_back(img);
	auto m1result = model->run(imgs);

}

void trTest()
{
	TaskThread task_thread;
	task_thread.createThreadPool(3);
	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
	TRConfig trconfig;
	trconfig.memory = 6;
	trconfig.modelPath = "D:\\TEST_DATA\\model\\uff\\model1WithoutSoftmax.uff";
	trconfig.inputName = "input_1";
	trconfig.outputName.emplace_back("dense_2/Sigmoid");
	trconfig.outputName.emplace_back("conv2d_1/BiasAdd");
	trconfig.outputName.emplace_back("softmax/output");
	trconfig.height = 512;
	trconfig.width = 512;
	trconfig.channel = 3;
	trconfig.batchsize = 1;
	trconfig.outputSize.emplace_back(1);
	trconfig.outputSize.emplace_back(1);
	trconfig.outputSize.emplace_back(512);

	auto model = get(new TRModel1(trconfig));
	cv::Mat img = cv::imread("D:\\TEST_OUTPUT\\rnnPredict\\sfy1148589 0893178\\model2\\0_114139_64427_0.999979.tif");
	cv::resize(img, img, cv::Size(512, 512));

	std::vector<cv::Mat> imgs;
	imgs.emplace_back(img);
	auto m1result = model->run(imgs);

	std::cout << "test\n";
}

void resnetTFTest()
{
	TaskThread task_thread;
	task_thread.createThreadPool(3);
	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
	TFConfig tfconfig;
	tfconfig.modelPath = "D:\\TEST_DATA\\model\\liusibo\\20201127\\model\\L2BD_model2_272_encoder.pb";
	tfconfig.inputName = "input_1:0";
	tfconfig.outputName.emplace_back("dense_2/Sigmoid:0");
	tfconfig.height = 256;
	tfconfig.width = 256;

	auto model = get(new Resnet50TF(tfconfig));
	cv::Mat img = cv::imread("D:\\TEST_OUTPUT\\rnnPredict\\sfy1148589 0893178\\model2\\0_114139_64427_0.999979.tif");
	cv::resize(img, img, cv::Size(256, 256));

	std::vector<cv::Mat> imgs;
	imgs.emplace_back(img);
	auto result = model->run(imgs);
	std::cout << "test" << std::endl;
}

void wholeSlideTest()
{
	setIniPath("./config.ini");
	std::cout << IniConfig::instance().getIniInt("ModelInput", "height") << std::endl;
	TaskThread task_thread;
	task_thread.createThreadPool(8);
	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
	TFConfig tfconfig;
	tfconfig.modelPath = "D:\\TEST_DATA\\model\\liusibo\\20201127\\model\\L2BD_model2_272_encoder.pb";
	tfconfig.inputName = "input_1:0";
	tfconfig.outputName.emplace_back("dense_2/Sigmoid:0");
	tfconfig.height = 256;
	tfconfig.width = 256;
	SlidePredict<
		tensorflow::Tensor,
		std::vector<tensorflow::Tensor>,
		cv::Mat,
		ResnetDST> slidePredict(new Resnet50TF(tfconfig));
	slidePredict.modelHeight = 256;
	slidePredict.modelWidth = 256;
	slidePredict.modelMpp = 0.293f;
	MultiImageRead mImgRead("D:\\TEST_DATA\\rnnPredict\\052800092.srp");
	mImgRead.createReadHandle(2);
	auto results = slidePredict.run(mImgRead);
	//取前十个保存图像
	for (int i = 0; i < 10; i++)
	{
		cv::Mat img;
		auto result = results[i];
		mImgRead.getTile(0, result.first.x, result.first.y, result.first.width, result.first.height, img);
		cv::imwrite(std::to_string(i) + "_" + std::to_string(result.second) + ".tif", img);
	}
}


template <typename MLIN, typename MLOUT, typename SRC, typename DST>
auto runWithSlidePredict(SlidePredict<MLIN, MLOUT, SRC, DST>* slidePredict, MultiImageRead &mImgRead)
{
	auto result = slidePredict->run(mImgRead);
	return result;
}

//void testSlidePredict()
//{
//	setIniPath("./config.ini");
//	std::cout << IniConfig::instance().getIniInt("ModelInput", "height") << std::endl;
//	TaskThread task_thread;
//	task_thread.createThreadPool(8);
//	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
//	TFConfig tfconfig;
//	MultiImageRead mImgRead("D:\\TEST_DATA\\rnnPredict\\052800092.srp");
//	mImgRead.createReadHandle(2);
//	tfconfig.modelPath = "D:\\TEST_DATA\\model\\liusibo\\20201127\\model\\L2BD_model2_272_encoder.pb";
//	tfconfig.inputName = "input_1:0";
//	tfconfig.outputName.emplace_back("dense_2/Sigmoid:0");
//	tfconfig.height = 256;
//	tfconfig.width = 256;
//	auto slidePredict = getSlidePredict(new Resnet50TF(tfconfig));
//	//auto result = slidePredict->run(mImgRead);
//	auto result = runWithSlidePredict(slidePredict, mImgRead);
//	
//}
//
//
//void testResnet50TR()
//{
//	setIniPath("./config.ini");
//	TaskThread taskThread;
//	taskThread.createThreadPool(8);
//	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
//	TRConfig trConfig("./config.ini");
//	MultiImageRead mImgRead("D:\\TEST_DATA\\rnnPredict\\052800092.srp");
//	mImgRead.createReadHandle(2);
//	auto slidePredict = getSlidePredict(new Resnet50TR(trConfig));
//	slidePredict->modelHeight = 256;
//	slidePredict->modelWidth = 256;
//	slidePredict->modelMpp = 0.293f;
//	auto result = runWithSlidePredict(slidePredict, mImgRead);
//	std::cout << "111" << std::endl;
//}
