//测试一下读图的效率对比

#include "SlidePredict.h"
#include "IniConfig.h"
#include "Resnet50TF.h"
#include "Resnet50TR.h"

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
extern SlidePredict<MLIN, MLOUT, SRC, DST>* getSlidePredict(Model<MLIN, MLOUT, SRC, DST>* model);

void testResnet50TR()
{
	setIniPath("./config.ini");
	TaskThread taskThread;
	taskThread.createThreadPool(IniConfig::instance().getIniInt("Thread", "Total"));
	_putenv_s("CUDA_VISIBLE_DEVICES", "0");
	TRConfig trConfig("./config.ini");
	std::string testSlide = IniConfig::instance().getIniString("ReadTest", "testSlide");
	MultiImageRead mImgRead(testSlide.c_str());
	mImgRead.createReadHandle(IniConfig::instance().getIniInt("Thread", "MultiImageRead"));
	auto slidePredict = getSlidePredict(new Resnet50TR(trConfig));
	slidePredict->modelHeight = IniConfig::instance().getIniInt("ModelInput", "height");
	slidePredict->modelWidth = IniConfig::instance().getIniInt("ModelInput", "width");
	slidePredict->modelMpp = IniConfig::instance().getIniDouble("ModelInput", "mpp");
	slidePredict->run(mImgRead);
	delete slidePredict;
}

//
//int main() {
//	testResnet50TR();
//	system("pause");
//	return 0;
//}