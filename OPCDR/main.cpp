#include <iostream>

#include <windows.h>
#include "interface.h"
#include <string>
#include "cuda_runtime_api.h"
#include <thread>
#include "AnnoRecommend.h"
#include "ModelRecommend.h"
#include "Slides.h"
#include "IniConfig.h"
#include "opencv2/opencv.hpp"
#include "MultiImageRead.h"

extern void wholeSlideTest();
extern void run2(Anno* annos, const char* ini_path, const char* slide, int recom_num, const char* savePath);



//ֻ����һ��mid�µ���Ƭ����Ϊ���mid����Ҫ���ģ�ͣ�����tf�����ԣ�ֻ�ܼ���һ��ģ��
void computeSlideInDatabase(std::string ini_path)
{
	//0.��ʼ����Դ
	setIniPath(ini_path);
	std::string gpu_id = IniConfig::instance().getIniString("Gpu", "id");
	setCudaVisibleDevices(gpu_id.c_str());

	std::string mid = IniConfig::instance().getIniString("Para", "mid");
	int recom_num = IniConfig::instance().getIniInt("Para", "recom_num");

	std::string slidePath = IniConfig::instance().getIniString("Slide", "path");
	std::string savePath = IniConfig::instance().getIniString("Save", "path");
	//���ݿ���Դ
	std::string hostName = IniConfig::instance().getIniString("Database", "hostName");
	std::string userName = IniConfig::instance().getIniString("Database", "userName");
	std::string password = IniConfig::instance().getIniString("Database", "password");
	ModelRecommend modelRecommend(hostName, userName, password);
	Slides slides(hostName, userName, password);
	AnnoRecommend annoRecommend(hostName, userName, password);

	//1.��ѯ��Ҫ�������Ƭ

	auto modelRecommendAttrs = modelRecommend.getModelUnFinishedSlide(mid);
	std::vector<SlidesAttr> uncompute_slides;
	for (auto attr : modelRecommendAttrs)
	{
		auto slidesAttr = slides.getSlideBySid(attr.sid);
		uncompute_slides.emplace_back(slidesAttr);
	}

	//2.ʹ��interface�еĺ�����ʼ����
	JavaHandle handle = initialize_handle(ini_path.c_str());
	int count = 0;
	for (auto elem : uncompute_slides)
	{
		Anno* annos = new Anno[recom_num];
		MyPoint* topLeft = new MyPoint[recom_num];
		std::string realSavePath = savePath + "\\" + mid + "\\" + std::to_string(elem.sid);
		std::string temp_slide_path = slidePath + "\\" + elem.slide_name;

		run(handle, annos, topLeft, ini_path.c_str(),  temp_slide_path.c_str(), recom_num, realSavePath.c_str());
		//run(handle, annos, ini_path.c_str(), (elem.slide_path + "/" + elem.slide_name).c_str(), recom_num, savePath.c_str());
		//��annosд�뵽���ݿ���
		for (int i = 0; i < recom_num; i++)
		{
			AnnoRecommendAttr attr;
			attr.sid = elem.sid;
			attr.mid = mid;
			attr.anno_class = "HSIL";
			attr.center_point = std::to_string(annos[i].x) + "," + std::to_string(annos[i].y);
			attr.top_left = std::to_string(topLeft[i].x) + "," + std::to_string(annos[i].y);
			
			annoRecommend.insert(attr);
		}
		//��״̬���µ�ModelRecommend����
		ModelRecommendAttr attr;
		attr.id = modelRecommendAttrs[count].id;
		attr.finished = "T";
		modelRecommend.updateStatus(attr);
		delete[] annos;
		count++;
	}
	freeModelMem(handle);
}

//extern void testResnet50TR();
int main4() 
{
	//����һ��resnet50 tr�汾�Ƿ����
	//testResnet50TR();
	system("pause");
	return 0;
}

extern void computeSlideInDatabase2(std::string iniPath);

//ֱ�������߳̽��м������ݿ��������Ƭ
int main(int argc, char** argv)
{
	using namespace std;
	std::string config_path = "";
	switch (argc) {
	case 1:
		config_path = "./config.ini";
		break;
	case 2:
		config_path = argv[1];
		break;
	default:
		std::cout << "please check your parameter\n";
	}
	computeSlideInDatabase2(config_path);
	//system("pause");
	return 0;
}

int main2()
{
	//wholeSlideTest();

	//����interface����ĺ������в���
	using namespace std;
	string ini_path = "./config.ini";
	string slide_path = "D:\\TEST_DATA\\rnnPredict\\052800092.srp";
	setCudaVisibleDevices("0");
	//JavaHandle handle = initialize_handle(ini_path.c_str());
	int recom_num = 10;
	Anno* annos = new Anno[10];
	std::string savePath = "D:\\TEST_OUTPUT\\Project\\C++\\OPCDR\\";

	std::thread t1(run2, annos, ini_path.c_str(), slide_path.c_str(), recom_num, savePath.c_str());

	t1.join();
	for (int i = 0; i < 10; i++)
	{
		std::cout << annos[i].score << " ";
	}
	//run(handle, annos, ini_path.c_str(), slide_path.c_str(), recom_num, savePath.c_str());
	//freeModelMem(handle);
	//cudaDeviceReset();
	system("pause");
	return 0;
}