#include <chrono>
#include "IniConfig.h"
#include "Resnet50TF.h"
#include "Resnet50TR.h"
#include "CommonFunction.h"
#include "Model.h"
#include <fstream>


template <typename MLIN, typename MLOUT, typename SRC, typename DST>
Model<MLIN, MLOUT, SRC, DST>* get2(Model<MLIN, MLOUT, SRC, DST>* model)
{
	return model;
}


template <typename DST>
void writeScore2Txt(std::string filename, std::vector<DST> result)
{
	std::string value;
	for (auto elem : result)
	{
		value = value + std::to_string(elem.getScore()) + "\n";
	}
	std::ofstream txtOut;
	txtOut.open(filename);
	txtOut << value;
	txtOut.close();
}

extern void tfTest();

void testModelEfficiency()
{
	using namespace std::chrono;


	std::string iniPath = "./config.ini";
	setIniPath(iniPath);

	_putenv_s("CUDA_VISIBLE_DEVICES", "0");



	TaskThread taskThread;
	int totalThreadNum = IniConfig::instance().getIniInt("Thread", "Total");
	taskThread.createThreadPool(totalThreadNum);


	std::string imgPath = IniConfig::instance().getIniString("Test", "imgPath");
	std::vector<std::string> imgPaths;

	getFiles(imgPath, imgPaths, "jpg");

	std::vector<cv::Mat> imgs;
	std::vector<cv::Mat> imgs2;
	for (auto elem : imgPaths) {
		cv::Mat img = cv::imread(elem);
		imgs.emplace_back(std::move(img));
	}

	int batchsize = IniConfig::instance().getIniInt("ModelInput", "batchsize");
	for (int i = 0; i < batchsize; i++)
	{
		imgs2.emplace_back(imgs[i].clone());
	}

	std::cout << "图像读取完成" << std::endl;

	std::string filename = "";
	if (IniConfig::instance().getIniString("TR", "use") == "ON")
	{

		TRConfig trConfig(iniPath.c_str());
		auto model = get2(new Resnet50TR(trConfig));
		model->run(imgs2);

		auto start = system_clock::now();
		auto results = model->run(imgs);


		auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
		cout << "花费了"
			<< double(duration.count()) * microseconds::period::num / microseconds::period::den
			<< "秒" << endl;

		if (IniConfig::instance().getIniString("TR", "quantize") == "ON")
		{
			filename = "quantize_result.txt";
		}
		else {
			filename = "tensorrt.txt";
		}
		writeScore2Txt(filename, results);
		delete model;
	}
	else {
		TFConfig tfConfig(iniPath.c_str());
		auto model = get2(new Resnet50TF(tfConfig));
		model->run(imgs2);

		auto start = system_clock::now();
		auto results = model->run(imgs);
		auto end = system_clock::now();
		auto duration = duration_cast<microseconds>(end - start);
		cout << "花费了"
			<< double(duration.count()) * microseconds::period::num / microseconds::period::den
			<< "秒" << endl;

		filename = "tensorflow.txt";
		writeScore2Txt(filename, results);
	}


	//for (auto elem : results)
	//{
	//	std::cout << elem << std::endl;
	//}
}
//
//int main()
//{
//	//tfTest();
//	testModelEfficiency();
//	system("pause");
//	return 0;
//}