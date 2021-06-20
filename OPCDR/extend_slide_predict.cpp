/*
 *利用模板使用SlidePredict，使得SlidePredict用途更加广泛
*/

#include "SlidePredict.h"
#include "IniConfig.h"
#include "Resnet50TF.h"
#include "Resnet50TR.h"
#include "MobileNetTF.h"
#include "MobileNetTR.h"
#include "interface.h"
#include "AnnoRecommend.h"
#include "Slides.h"
#include "ModelRecommend.h"
#include "TrainConfig.h"
#include "ModelConfig.h"
#include "ModelList.h"

std::string getTROutByTFOut(std::string TFOut)
{
	auto splitTFOut = split(TFOut, ',');
	std::vector<std::string> splitTROut;
	for (auto elem : splitTFOut)
	{
		std::string singleTROut;
		auto temp = split(elem, ':');
		singleTROut = temp[0];
		splitTROut.emplace_back(singleTROut);
	}
	std::string TROut;
	for (auto elem : splitTROut)
	{
		TROut = TROut + elem + ',';
	}
	TROut.pop_back();
	return TROut;
}

std::string getTRInByTFIn(std::string TFIn)
{
	auto temp = split(TFIn, ':');
	return temp[0];
}

//根据数据库来初始化ini文件中的内容（为了不修改以前那么多的代码，只能修改ini文件了）
void updateIniByDatabase()
{
	//如果仅本地测试，则不更新
	if (IniConfig::instance().getIniString("Local", "flag") == "ON") {
		return;
	}
	auto trainID = IniConfig::instance().getIniString("Para", "trainid");
	auto hostName = IniConfig::instance().getIniString("Database", "hostName");
	auto userName = IniConfig::instance().getIniString("Database", "userName");
	auto password = IniConfig::instance().getIniString("Database", "password");

	//1.修改模型的宽高、mpp
	TrainConfig trainConfig(hostName, userName, password);
	auto trainConfigAttr = trainConfig.getTrainConfigAttrByTrainId(trainID);

	//2.修改tensorflow和tensorrt模型的输入输出和模型路径
	//修改输入输出
	auto modelConfigID = trainConfigAttr.modelconfigid;
	ModelConfig modelConfig(hostName, userName, password);
	auto modelConfigAttr = modelConfig.getAttrByID(modelConfigID);
	std::string modelname = modelConfigAttr.modelname;
	ModelList modelList(hostName, userName, password);
	//auto temp_results = modelList.getAttrByModelName(modelname);
	auto modelListAttr = modelList.getAttrByModelName(modelname);
	IniConfig::instance().setIniString("Para", "modelname", modelname);
	auto input = modelListAttr.input;
	auto output = modelListAttr.output;
	auto TRIn = getTRInByTFIn(input);
	auto TROut = getTROutByTFOut(output);
	if (modelname == "resnet50") {
		IniConfig::instance().setIniString("Resnet50", "height", trainConfigAttr.dst_size);
		IniConfig::instance().setIniString("Resnet50", "width", trainConfigAttr.dst_size);
		IniConfig::instance().setIniString("Resnet50", "mpp", trainConfigAttr.dst_mpp);

		IniConfig::instance().setIniString("Resnet50TF", "input", input);
		IniConfig::instance().setIniString("Resnet50TF", "output", output);
		IniConfig::instance().setIniString("Resnet50TR", "input", TRIn);
		IniConfig::instance().setIniString("Resnet50TR", "output", TROut);
		IniConfig::instance().setIniString("Resnet50TR", "output_size", modelListAttr.output_size);
		//修改模型路径
		IniConfig::instance().setIniString("Resnet50TF", "path", trainConfigAttr.modelpath);
		IniConfig::instance().setIniString("Resnet50TR", "path", trainConfigAttr.uffpath);
	}
	if (modelname == "mobilenet") {
		IniConfig::instance().setIniString("MobileNet", "height", trainConfigAttr.dst_size);
		IniConfig::instance().setIniString("MobileNet", "width", trainConfigAttr.dst_size);
		IniConfig::instance().setIniString("MobileNet", "mpp", trainConfigAttr.dst_mpp);

		IniConfig::instance().setIniString("MobileNetTF", "input", input);
		IniConfig::instance().setIniString("MobileNetTF", "output", output);
		IniConfig::instance().setIniString("MobileNetTR", "input", TRIn);
		IniConfig::instance().setIniString("MobileNetTR", "output", TROut);
		IniConfig::instance().setIniString("MobileNetTR", "output_size", modelListAttr.output_size);
		//修改模型路径
		IniConfig::instance().setIniString("MobileNetTF", "path", trainConfigAttr.modelpath);
		IniConfig::instance().setIniString("MobileNetTR", "path", trainConfigAttr.uffpath);
	}
}


//根据泛型Model生成泛型的SlidePredict
template <typename MLIN, typename MLOUT, typename SRC, typename DST>
SlidePredict<MLIN, MLOUT, SRC, DST>* getSlidePredict(Model<MLIN, MLOUT, SRC, DST>* model)
{
	return new SlidePredict<MLIN, MLOUT, SRC, DST>(model);
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
void initializeSlidePredict(SlidePredict<MLIN, MLOUT, SRC, DST>* slidePredict)
{
	std::string modelname = IniConfig::instance().getIniString("Para", "modelname");
	if (modelname == "resnet50") {
		slidePredict->modelHeight = IniConfig::instance().getIniInt("Resnet50", "height");
		slidePredict->modelWidth = IniConfig::instance().getIniInt("Resnet50", "width");
		slidePredict->modelMpp = IniConfig::instance().getIniDouble("Resnet50", "mpp");
	}
	if (modelname == "mobilenet") {
		slidePredict->modelHeight = IniConfig::instance().getIniInt("MobileNet", "height");
		slidePredict->modelWidth = IniConfig::instance().getIniInt("MobileNet", "width");
		slidePredict->modelMpp = IniConfig::instance().getIniDouble("MobileNet", "mpp");
	}
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
SlidePredict<MLIN, MLOUT, SRC, DST>* getSlidePredict(std::string iniPath)
{
	std::string modelname = IniConfig::instance().getIniString("Para", "modelname");
	std::string useTR = IniConfig::instance().getIniString("TensorRT", "use");
	TFConfig tfConfig(iniPath.c_str(), modelname);
	TRConfig trConfig(iniPath.c_str(), modelname);
	if (modelname == "resnet50") {
		if (useTR == "ON") {
			return getSlidePredict(new Resnet50TR(trConfig));
		}
		else {
			return getSlidePredict(new Resnet50TF(tfConfig));
		}
	}
	if (modelname == "modelname") {
		if (useTR == "ON") {
			return getSlidePredict(new MobileNetTR(trConfig));
		}
		else {
			return getSlidePredict(new MobileNetTF(tfConfig));
		}
	}
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
//template <typename MLIN, typename MLOUT, typename SRC, typename DST>
//SlidePredict<MLIN, MLOUT, SRC, DST>* getSlidePredict(const char* iniPath)
//{
//	if (IniConfig::instance().getIniString("TR", "use") == "ON")
//	{
//		TRConfig trConfig(iniPath);
//		auto slidePredict = getSlidePredict(new Resnet50TR(trConfig));
//		return slidePredict;
//	}
//	else 
//	{
//		TFConfig tfConfig(iniPath);
//		auto slidePredict = getSlidePredict(new Resnet50TF(tfConfig));
//		return slidePredict;
//	}
//}

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
	std::string trainid,
	int recom_num)
{
	int count = 0;
	for (auto elem : uncompute_slides)
	{
		std::string realSavePath = savePath + "\\" + trainid + "\\" + std::to_string(elem.sid);

		//std::string temp_slide_path = slidePath + "\\" + elem.slide_name;
		std::string temp_slide_path;
		if (IniConfig::instance().getIniString("Local", "flag") == "ON") {
			temp_slide_path = slidePath + "\\" + elem.slide_name;
		}
		else {
			temp_slide_path = elem.slide_path + "//" + elem.slide_name;
		}
		std::cout << "running slide: " << temp_slide_path << std::endl;

		MultiImageRead mImgRead(temp_slide_path.c_str());
		createDirRecursive(realSavePath);
		mImgRead.createReadHandle(2);
		auto results = slidePredict->run(mImgRead);

		int thumbnail_height = slidePredict->thumbnail.rows;
		int thumbnail_width = slidePredict->thumbnail.cols;
		float ratio = (float)slidePredict->slideHeight / (float)thumbnail_height;

		for (int i = 0; i < recom_num; i++)
		{
			auto result = results[i];
			AnnoRecommendAttr attr;
			attr.sid = elem.sid;
			attr.mid = trainid;
			//attr.anno_class = "HSIL";
			attr.anno_class = result.second.getType();
			attr.center_point = std::to_string(result.first.x + result.first.width / 2) + "," + std::to_string(result.first.y + result.first.height / 2);
			cv::Mat img;
			//为了使医生能够更好地判断，因此，裁取更大一点的范围(例如两倍)
			
			//扩充大小
			int expansionHeight = 500;
			int expansionWidth = 500;
			cv::Point newPoint(result.first.x - expansionWidth, result.first.y - expansionHeight);
			if (newPoint.x < 0 || newPoint.y < 0) {
				//如果范围超过了边界，保存原图
				mImgRead.getTile(0, result.first.x, result.first.y, result.first.width, result.first.height, img);
				cv::imwrite(realSavePath + "\\" + std::to_string(i) + ".jpg", img);
				attr.top_left = std::to_string(result.first.x) + "," + std::to_string(result.first.y);
			}
			else {
				mImgRead.getTile(0, newPoint.x, newPoint.y, 
					result.first.width + expansionWidth* 2, 
					result.first.height  + expansionHeight * 2, img);
				cv::rectangle(
					img, cv::Rect(
						expansionWidth,
						expansionHeight,
						result.first.width,
						result.first.height),
					cv::Scalar(0, 0, 255), 
					4);
				cv::imwrite(realSavePath + "\\" + std::to_string(i) + ".jpg", img);
				attr.top_left = std::to_string(newPoint.x) + "," + std::to_string(newPoint.y);
			}
			//在这里保存原始图像
			mImgRead.getTile(0, result.first.x, result.first.y, result.first.width, result.first.height, img);
			cv::imwrite(realSavePath + "\\" + std::to_string(i) + "_src.jpg", img);


			annoRecommend.insert(attr);

			int temp_x = (float)result.first.x / ratio;
			int temp_y = (float)result.first.y / ratio;
			int radius = (100.0f / ratio) / slidePredict->slideMpp;
			cv::circle(slidePredict->thumbnail, cv::Point(temp_x, temp_y), radius, cv::Scalar(0, 0, 255), 4);
		}
		cv::imwrite(realSavePath + "\\thumbnail.jpg", slidePredict->thumbnail);
		ModelRecommendAttr attr;
		attr.id = modelRecommendAttrs[count].id;
		attr.finished = "T";
		modelRecommend.updateStatus(attr);
		count++;
	}
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
void runWithSlidePredict(
	SlidePredict<MLIN, MLOUT, SRC, DST>* slidePredict, 
	std::vector<SlidesAttr> &uncompute_slides,
	AnnoRecommend &annoRecommend,
	ModelRecommend &modelRecommend,
	ModelRecommendAttr &modelRecommendAttrs,
	std::string savePath,
	std::string slidePath,
	std::string trainid,
	int recom_num)
{
	initializeSlidePredict(slidePredict);
	computeSlides(
		slidePredict, 
		uncompute_slides, 
		annoRecommend, 
		modelRecommend, 
		modelRecommendAttrs, savePath, slidePath, trainid, recom_num);
	delete slidePredict;
}

void computeSlideInDatabase2(std::string iniPath)
{
	std::cout << "computeSlideInDatabase2" << std::endl;
	setIniPath(iniPath);
	updateIniByDatabase();
	std::string gpu_id = IniConfig::instance().getIniString("Gpu", "id");
	setCudaVisibleDevices(gpu_id.c_str());
	std::cout << "set gpu:" << gpu_id << std::endl;

	std::string trainid = IniConfig::instance().getIniString("Para", "trainid");
	std::cout << "train id: " << trainid << std::endl;
	int recom_num = IniConfig::instance().getIniInt("Para", "recom_num");

	std::string slidePath = IniConfig::instance().getIniString("Slide", "path");
	std::cout << "slidePath:" << slidePath << std::endl;
	std::string savePath = IniConfig::instance().getIniString("Save", "path");
	std::cout << "opcdr recommend save path:" << savePath << std::endl;
	//数据库资源
	std::string hostName = IniConfig::instance().getIniString("Database", "hostName");
	std::cout << "database hostname:" << hostName << std::endl;
	std::string userName = IniConfig::instance().getIniString("Database", "userName");
	std::cout << "database username:" << userName << std::endl;
	std::string password = IniConfig::instance().getIniString("Database", "password");
	std::cout << "database password:" << password << std::endl;
	ModelRecommend modelRecommend(hostName, userName, password);
	Slides slides(hostName, userName, password);
	AnnoRecommend annoRecommend(hostName, userName, password);

	//1.查询需要计算的切片

	auto modelRecommendAttrs = modelRecommend.getModelUnFinishedSlide(trainid);
	std::vector<SlidesAttr> uncompute_slides;
	for (auto attr : modelRecommendAttrs)
	{
		auto slidesAttr = slides.getSlideBySid(attr.sid);
		uncompute_slides.emplace_back(slidesAttr);
	}

	std::cout << "uncompute slides size: " << uncompute_slides.size() << std::endl;

	TaskThread taskThread;
	taskThread.createThreadPool(8);

	std::string modelname = IniConfig::instance().getIniString("Para", "modelname");
	std::string useTR = IniConfig::instance().getIniString("TensorRT", "use");
	TFConfig tfConfig(iniPath.c_str(), modelname);
	TRConfig trConfig(iniPath.c_str(), modelname);
	if (modelname == "resnet50") {
		if (useTR == "ON") {
			auto slidePredict = getSlidePredict(new Resnet50TR(trConfig));
			//runWithSlidePredict(slidePredict, uncompute_slides,
			//	annoRecommend,
			//	modelRecommend,
			//	modelRecommendAttrs, savePath, slidePath, trainid, recom_num);
			initializeSlidePredict(slidePredict);
			computeSlides(
				slidePredict,
				uncompute_slides,
				annoRecommend,
				modelRecommend,
				modelRecommendAttrs, savePath, slidePath, trainid, recom_num);
			delete slidePredict;
		}
		else {
			auto slidePredict = getSlidePredict(new Resnet50TF(tfConfig));
			initializeSlidePredict(slidePredict);
			computeSlides(
				slidePredict,
				uncompute_slides,
				annoRecommend,
				modelRecommend,
				modelRecommendAttrs, savePath, slidePath, trainid, recom_num);
			delete slidePredict;
		}
	}
	if (modelname == "mobilenet") {
		if (useTR == "ON") {
			auto slidePredict = getSlidePredict(new MobileNetTR(trConfig));
			initializeSlidePredict(slidePredict);
			computeSlides(
				slidePredict,
				uncompute_slides,
				annoRecommend,
				modelRecommend,
				modelRecommendAttrs, savePath, slidePath, trainid, recom_num);
			delete slidePredict;
		}
		else {
			auto slidePredict = getSlidePredict(new MobileNetTF(tfConfig));
			initializeSlidePredict(slidePredict);
			computeSlides(
				slidePredict,
				uncompute_slides,
				annoRecommend,
				modelRecommend,
				modelRecommendAttrs, savePath, slidePath, trainid, recom_num);
			delete slidePredict;
		}
	}

	//auto slidePredict = getSlidePredict(iniPath);

	//if (IniConfig::instance().getIniString("TR", "use") == "ON")
	//{
	//	TRConfig trConfig(iniPath.c_str());
	//	auto slidePredict = getSlidePredict(new Resnet50TR(trConfig));
	//	initializeSlidePredict(slidePredict);
	//	computeSlides(slidePredict, uncompute_slides, annoRecommend, modelRecommend, modelRecommendAttrs, savePath, slidePath, trainid, recom_num);
	//	delete slidePredict;
	//}
	//else
	//{
	//	TFConfig tfConfig(iniPath.c_str());
	//	auto slidePredict = getSlidePredict(new Resnet50TF(tfConfig));
	//	initializeSlidePredict(slidePredict);
	//	computeSlides(slidePredict, uncompute_slides, annoRecommend, modelRecommend, modelRecommendAttrs, savePath, slidePath, trainid, recom_num);
	//	delete slidePredict;
	//}
}