#include "interface.h"
#include "SlidePredict.h"
#include "Resnet50TF.h"


class JavaResource 
{
public:
	SlidePredict<
		tensorflow::Tensor,
		std::vector<tensorflow::Tensor>,
		cv::Mat,
		ResnetDST>* slidePredict = nullptr;
	TaskThread* taskThread = nullptr;
};

//使用run2的时候开启线程来执行，那么就没大问题了应该(没用的，只要tensorflow进程不退出，就永远不会释放显存)
void run2(Anno* annos, const char* ini_path, const char* slide, int recom_num, const char* savePath)
{
	setIniPath(ini_path);

	TFConfig tfconfig(ini_path);

	auto slidePredict = new SlidePredict<
		tensorflow::Tensor,
		std::vector<tensorflow::Tensor>,
		cv::Mat,
		ResnetDST>(new Resnet50TF(tfconfig));

	JavaResource* resource = new JavaResource();
	resource->slidePredict = slidePredict;
	resource->taskThread = new TaskThread();
	resource->taskThread->createThreadPool(8);

	slidePredict->modelHeight = IniConfig::instance().getIniInt("ModelInput", "height");
	slidePredict->modelWidth = IniConfig::instance().getIniInt("ModelInput", "width");
	slidePredict->modelMpp = IniConfig::instance().getIniDouble("ModelInput", "mpp");



	std::string savePathStr = std::string(savePath);
	createDirRecursive(savePath);

	//auto slidePredict = resource->slidePredict;
	MultiImageRead mImgRead(slide);
	mImgRead.createReadHandle(2);

	auto results = slidePredict->run(mImgRead);
	for (int i = 0; i < recom_num; i++)
	{
		cv::Mat img;
		auto result = results[i];
		annos[i].id = i;
		annos[i].x = result.first.x + result.first.width / 2;
		annos[i].y = result.first.y + result.first.height / 2;
		annos[i].score = result.second;
		mImgRead.getTile(0, result.first.x, result.first.y, result.first.width, result.first.height, img);
		cv::imwrite(savePathStr + "\\" + std::to_string(i) + ".jpg", img);
	}
	//在保存一下缩略图
	cv::imwrite(savePathStr + "\\thumbnail.jpg", slidePredict->thumbnail);



	//auto resource = (JavaResource*)handle;
	delete resource->slidePredict;
	resource->slidePredict = nullptr;
	delete resource->taskThread;
	resource->taskThread = nullptr;
	delete resource;
	resource = nullptr;

}

JavaHandle initialize_handle(const char* ini_path)
{
	setIniPath(ini_path);

	TFConfig tfconfig(ini_path);

	auto slidePredict = new SlidePredict<
		tensorflow::Tensor,
		std::vector<tensorflow::Tensor>,
		cv::Mat,
		ResnetDST>(new Resnet50TF(tfconfig));

	JavaResource* resource = new JavaResource();
	resource->slidePredict = slidePredict;
	resource->taskThread = new TaskThread();
	resource->taskThread->createThreadPool(8);

	slidePredict->modelHeight = IniConfig::instance().getIniInt("ModelInput", "height");
	slidePredict->modelWidth = IniConfig::instance().getIniInt("ModelInput", "width");
	slidePredict->modelMpp = IniConfig::instance().getIniDouble("ModelInput", "mpp");

	return JavaHandle(resource);
}

void run(JavaHandle handle, Anno* annos, MyPoint *topLeft, const char* ini_path, const char* slide, int recom_num, const char *savePath)
{
	//std::string savePath = IniConfig::instance().getIniString("Save", "path");
	std::string savePathStr = std::string(savePath);
	createDirRecursive(savePath);

	auto resource = (JavaResource*)handle;
	auto slidePredict = resource->slidePredict;
	MultiImageRead mImgRead(slide);
	mImgRead.createReadHandle(2);

	auto results = slidePredict->run(mImgRead);
	for (int i = 0; i < recom_num; i++)
	{
		cv::Mat img;
		auto result = results[i];
		annos[i].id = i;
		annos[i].x = result.first.x + result.first.width / 2;
		annos[i].y = result.first.y + result.first.height / 2;
		annos[i].score = result.second;
		topLeft[i].x = result.first.x;
		topLeft[i].y = result.first.y;
		mImgRead.getTile(0, result.first.x, result.first.y, result.first.width, result.first.height, img);
		cv::imwrite(savePathStr + "\\" + std::to_string(i) + ".jpg", img);
	}
	//在保存一下缩略图
	cv::imwrite(savePathStr + "\\thumbnail.jpg", slidePredict->thumbnail);
}

void freeModelMem(JavaHandle handle)
{
	auto resource = (JavaResource*)handle;
	delete resource->slidePredict;
	resource->slidePredict = nullptr;
	delete resource->taskThread;
	resource->taskThread = nullptr;
	delete resource;
	resource = nullptr;

}

void setCudaVisibleDevices(const char* num)
{
	_putenv_s("CUDA_VISIBLE_DEVICES", num);
}

void setAdditionalPath(const char* path)
{
	std::string env = getenv("PATH");
	env += ";" + string(path);
	std::string newEnv = "PATH=" + env;
	_putenv(newEnv.c_str());
}