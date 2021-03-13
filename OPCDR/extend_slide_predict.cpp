/*
 *利用模板使用SlidePredict，使得SlidePredict用途更加广泛
*/

#include "SlidePredict.h"
#include "IniConfig.h"
#include "Resnet50TF.h"

#include "Resnet50TR.h"
#include "interface.h"
#include "AnnoRecommend.h"
#include "Slides.h"
#include "ModelRecommend.h"


//根据泛型Model生成泛型的SlidePredict
template <typename MLIN, typename MLOUT, typename SRC, typename DST>
SlidePredict<MLIN, MLOUT, SRC, DST>* getSlidePredict(Model<MLIN, MLOUT, SRC, DST>* model)
{
	return new SlidePredict<MLIN, MLOUT, SRC, DST>(model);
}

//以后可以将DST封装一下，里面都包含saveResult函数，将本身的结果存到数据库里面，这样的话就更完整了，但是现在的情况是没有整合，因此现在不写
//或者也可以传入一个函数，根据不同的DST传入不同的function(DST)
//这里就将就一下，专门处理Resnet50TF的结果吧
template<typename DST>
void saveResult(std::vector<cv::Rect, DST>& results, MultiImageRead& mImgRead, std::string iniPath)
{
	
}

//根据工厂模式生成SlidePredict(并不能广泛使用，是根据ini文件来决定生成tr还是tf)
//结论：不行，使用时还是需要传入模板参数
template <typename MLIN, typename MLOUT, typename SRC, typename DST>
SlidePredict<MLIN, MLOUT, SRC, DST>* getSlidePredict(const char* iniPath)
{
	if (IniConfig::instance().getIniString("TR", "use") == "ON")
	{
		TRConfig trConfig(iniPath);
		auto slidePredict = getSlidePredict(new Resnet50TR(trConfig));
		return slidePredict;
	}
	else 
	{
		TFConfig tfConfig(iniPath);
		auto slidePredict = getSlidePredict(new Resnet50TF(tfConfig));
		return slidePredict;
	}
}

void testGetSlidePredictByIni()
{
	std::string iniPath = "./config.ini";
	setIniPath(iniPath);
	//auto slidePredict = getSlidePredict(iniPath.c_str());
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
void computeSlides(
	SlidePredict<MLIN, MLOUT, SRC, DST>* slidePredict, 
	std::vector<SlidesAttr>& uncompute_slides, 
	AnnoRecommend& annoRecommend,
	ModelRecommend &modelRecommend,
	std::vector<ModelRecommendAttr> & modelRecommendAttrs,
	std::string savePath,
	std::string slidePath,
	std::string mid,
	int recom_num)
{
	int count = 0;
	for (auto elem : uncompute_slides)
	{
		std::string realSavePath = savePath + "\\" + mid + "\\" + std::to_string(elem.sid);

		std::string temp_slide_path = slidePath + "\\" + elem.slide_name;
		MultiImageRead mImgRead(temp_slide_path.c_str());
		createDirRecursive(realSavePath);
		mImgRead.createReadHandle(2);
		auto results = slidePredict->run(mImgRead);
		for (int i = 0; i < recom_num; i++)
		{
			auto result = results[i];
			AnnoRecommendAttr attr;
			attr.sid = elem.sid;
			attr.mid = mid;
			attr.anno_class = "HSIL";
			attr.center_point = std::to_string(result.first.x + result.first.width / 2) + "," + std::to_string(result.first.y + result.first.height / 2);
			attr.top_left = std::to_string(result.first.x) + "," + std::to_string(result.first.y);
			cv::Mat img;
			mImgRead.getTile(0, result.first.x, result.first.y, result.first.width, result.first.height, img);
			cv::imwrite(realSavePath + "\\" + std::to_string(i) + ".jpg", img);
			annoRecommend.insert(attr);
		}
		cv::imwrite(realSavePath + "\\thumbnail.jpg", slidePredict->thumbnail);
		ModelRecommendAttr attr;
		attr.id = modelRecommendAttrs[count].id;
		attr.finished = "T";
		modelRecommend.updateStatus(attr);
		count++;
	}
}

void computeSlideInDatabase2(std::string iniPath)
{
	setIniPath(iniPath);
	std::string gpu_id = IniConfig::instance().getIniString("Gpu", "id");
	setCudaVisibleDevices(gpu_id.c_str());

	std::string mid = IniConfig::instance().getIniString("Para", "mid");
	int recom_num = IniConfig::instance().getIniInt("Para", "recom_num");

	std::string slidePath = IniConfig::instance().getIniString("Slide", "path");
	std::string savePath = IniConfig::instance().getIniString("Save", "path");
	//数据库资源
	std::string hostName = IniConfig::instance().getIniString("Database", "hostName");
	std::string userName = IniConfig::instance().getIniString("Database", "userName");
	std::string password = IniConfig::instance().getIniString("Database", "password");
	ModelRecommend modelRecommend(hostName, userName, password);
	Slides slides(hostName, userName, password);
	AnnoRecommend annoRecommend(hostName, userName, password);

	//1.查询需要计算的切片

	auto modelRecommendAttrs = modelRecommend.getModelUnFinishedSlide(mid);
	std::vector<SlidesAttr> uncompute_slides;
	for (auto attr : modelRecommendAttrs)
	{
		auto slidesAttr = slides.getSlideBySid(attr.sid);
		uncompute_slides.emplace_back(slidesAttr);
	}

	TaskThread taskThread;
	taskThread.createThreadPool(8);

	if (IniConfig::instance().getIniString("TR", "use") == "ON")
	{
		TRConfig trConfig(iniPath.c_str());
		auto slidePredict = getSlidePredict(new Resnet50TR(trConfig));
		slidePredict->modelHeight = IniConfig::instance().getIniInt("ModelInput", "height");
		slidePredict->modelWidth = IniConfig::instance().getIniInt("ModelInput", "width");
		slidePredict->modelMpp = IniConfig::instance().getIniDouble("ModelInput", "mpp");
		computeSlides(slidePredict, uncompute_slides, annoRecommend, modelRecommend, modelRecommendAttrs, savePath, slidePath, mid, recom_num);
		delete slidePredict;
	}
	else
	{
		TFConfig tfConfig(iniPath.c_str());
		auto slidePredict = getSlidePredict(new Resnet50TF(tfConfig));
		slidePredict->modelHeight = IniConfig::instance().getIniInt("ModelInput", "height");
		slidePredict->modelWidth = IniConfig::instance().getIniInt("ModelInput", "width");
		slidePredict->modelMpp = IniConfig::instance().getIniDouble("ModelInput", "mpp");
		computeSlides(slidePredict, uncompute_slides, annoRecommend, modelRecommend, modelRecommendAttrs, savePath, slidePath, mid, recom_num);
		delete slidePredict;
	}
}