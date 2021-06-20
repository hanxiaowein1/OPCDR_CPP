#include <chrono>
#include "IniConfig.h"
#include "Resnet50TF.h"
#include "Resnet50TR.h"
#include "CommonFunction.h"
#include "SlidePredict.h"
#include "SlideRead.h"
#include "SlideFactory.h"
//#include ""

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
extern SlidePredict<MLIN, MLOUT, SRC, DST>* getSlidePredict(Model<MLIN, MLOUT, SRC, DST>* model);

void Mat2Tensor(std::vector<cv::Mat>& imgs, tensorflow::Tensor& tensor)
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


std::vector<cv::Rect> iniRects(
	int sHeight, int sWidth, int height, int width,
	int overlap, bool flag_right, bool flag_down,
	int& rows, int& cols)
{
	std::vector<cv::Rect> rects;
	//进行参数检查
	if (sHeight == 0 || sWidth == 0 || height == 0 || width == 0) {
		std::cout << "iniRects: parameter should not be zero\n";
		return rects;
	}
	if (sHeight > height || sWidth > width) {
		std::cout << "iniRects: sHeight or sWidth > height or width\n";
		return rects;
	}
	if (overlap >= sWidth || overlap >= height) {
		std::cout << "overlap should < sWidth or sHeight\n";
		return rects;
	}
	int x_num = (width - overlap) / (sWidth - overlap);
	int y_num = (height - overlap) / (sHeight - overlap);
	std::vector<int> xStart;
	std::vector<int> yStart;
	if ((x_num * (sWidth - overlap) + overlap) == width) {
		flag_right = false;
	}
	if ((y_num * (sHeight - overlap) + overlap) == height) {
		flag_down = false;
	}
	for (int i = 0; i < x_num; i++) {
		xStart.emplace_back((sWidth - overlap) * i);
	}
	for (int i = 0; i < y_num; i++) {
		yStart.emplace_back((sHeight - overlap) * i);
	}
	if (flag_right)
		xStart.emplace_back(width - sWidth);
	if (flag_down)
		yStart.emplace_back(height - sHeight);
	cols = xStart.size();
	rows = yStart.size();
	for (int i = 0; i < yStart.size(); i++) {
		for (int j = 0; j < xStart.size(); j++) {
			cv::Rect rect;
			rect.x = xStart[j];
			rect.y = yStart[i];
			rect.width = sWidth;
			rect.height = sHeight;
			rects.emplace_back(rect);
		}
	}
	return rects;
}

int main()
{
	//不断地从切片中读取30个区域，送入模型进行计算
	std::string iniPath = "./config.ini";
	setIniPath(iniPath);
	_putenv_s("CUDA_VISIBLE_DEVICES", "0");

	using namespace std::chrono;
	auto start = system_clock::now();

	std::string testSlide = IniConfig::instance().getIniString("ReadTest", "testSlide");
	std::unique_ptr<SlideFactory> sFactory(new SlideFactory());
	std::unique_ptr<SlideRead> sRead = sFactory->createSlideProduct(testSlide.c_str());
	int modelHeight = IniConfig::instance().getIniInt("ModelInput", "height");
	int modelWidth = IniConfig::instance().getIniInt("ModelInput", "width");
	double modelMpp = IniConfig::instance().getIniDouble("ModelInput", "mpp");
	std::string modelPath = IniConfig::instance().getIniString("Resnet50TF", "path");
	std::string inputName = "input_1:0";
	std::vector<std::string> outputName;
	outputName.emplace_back("dense_2/Sigmoid:0");
	outputName.emplace_back("global_max_pooling2d_1/Max:0");

	int slideHeight, slideWidth = 0;
	sRead->getSlideHeight(slideHeight);
	sRead->getSlideWidth(slideWidth);
	double slideMpp = 0.0f;
	sRead->getSlideMpp(slideMpp);
	int temp_cols = 0;
	int temp_rows = 0;
	std::vector<cv::Rect> rects2 = iniRects(
		modelHeight * float(modelMpp / slideMpp),
		modelWidth * float(modelMpp / slideMpp),
		slideHeight,
		slideWidth,
		(0.25f * modelHeight) * float(modelMpp / slideMpp),
		true,
		true,
		temp_rows,
		temp_cols
	);
	//要从头自己定义
	tensorflow::GraphDef graph_def;
	tensorflow::Status load_graph_status =
		ReadBinaryProto(tensorflow::Env::Default(),
			modelPath,
			&graph_def);
	if (!load_graph_status.ok()) {
		std::cout << modelPath << ": [LoadGraph] load graph failed!\n";
		return -1;
	}

	tensorflow::SessionOptions options;
	options.config.mutable_device_count()->insert({ "GPU",1 });
	options.config.mutable_gpu_options()->set_allow_growth(true);
	options.config.mutable_gpu_options()->set_force_gpu_compatible(true);
	std::unique_ptr<tensorflow::Session> m_session;
	m_session.reset(tensorflow::NewSession(options));
	auto status_creat_session = m_session.get()->Create(graph_def);
	std::cout << modelPath << "create session success\n";
	if (!status_creat_session.ok()) {
		std::cout << "[LoadGraph] creat session failed!\n" << std::endl;
		return -1;
	}

	for (int i = 0; i < rects2.size(); i= i + 30)
	{
		std::vector<cv::Mat> imgs;
		for (int j = i; (j < rects2.size()) && (j < i + 30); j++)
		{
			//不停地读取图像
			cv::Mat img;
			auto rect = rects2[j];
			sRead->getTile(0, rect.x, rect.y, rect.width, rect.height, img);
			cv::resize(img, img, cv::Size(modelWidth, modelHeight));
			imgs.emplace_back(std::move(img));
		}
		//然后开始转图像到Tensor
		int height = imgs[0].rows;
		int width = imgs[0].cols;
		int channel = imgs[0].channels();
		int size = imgs.size();
		tensorflow::Tensor in(tensorflow::DataType::DT_FLOAT,
			tensorflow::TensorShape({ size, height, width, channel }));
		Mat2Tensor(imgs, in);

		std::vector<tensorflow::Tensor> out;
		auto status_run = m_session->Run({ { inputName, in } },
			outputName, {}, &out);
		if (!status_run.ok()) {
			std::cout << "run model failed!\n";
		}

	}

	auto end = system_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	cout << "花费了"
		<< double(duration.count()) * microseconds::period::num / microseconds::period::den
		<< "秒" << endl;

	system("pause");
	return 0;
}


//测试模型的整体速度，测试不同的线程配比对比
int main_test_whole_speed_with_different_thread()
{
	using namespace std::chrono;
	std::string iniPath = "./config.ini";
	setIniPath(iniPath);
	_putenv_s("CUDA_VISIBLE_DEVICES", "0");

	TaskThread taskThread;
	int totalThreadNum = IniConfig::instance().getIniInt("Thread", "Total");
	taskThread.createThreadPool(totalThreadNum);

	std::string testSlide = IniConfig::instance().getIniString("ReadTest", "testSlide");
	MultiImageRead mImgRead(testSlide.c_str());
	mImgRead.createReadHandle(IniConfig::instance().getIniInt("Thread", "MultiImageRead"));

	TRConfig trConfig("./config.ini");
	auto slidePredict = getSlidePredict(new Resnet50TR(trConfig));
	slidePredict->modelHeight = IniConfig::instance().getIniInt("ModelInput", "height");
	slidePredict->modelWidth = IniConfig::instance().getIniInt("ModelInput", "width");
	slidePredict->modelMpp = IniConfig::instance().getIniDouble("ModelInput", "mpp");
	auto start = system_clock::now();
	slidePredict->run(mImgRead);
	auto end = system_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	cout << "花费了"
		<< double(duration.count()) * microseconds::period::num / microseconds::period::den
		<< "秒" << endl;
	delete slidePredict;

	system("pause");
	return 0;
}