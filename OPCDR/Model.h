#pragma once

#ifndef _OPCDR_MODEL_H_
#define _OPCDR_MODEL_H_

#include <functional>
#include <queue>
#include <mutex>
#include "TaskThread.h"
#include "PopQueueData.h"
#include "IniConfig.h"

/*
	描述基本模型的过程（输入→输出）
	IN:	模型的输入
	OUT:模型的输出
	SRC:模型的原始输入(cv::Mat等)
	DST:模型的目标输出(M1Result等)
*/

/// <summary>
/// 该类用来描述深度学习框架执行模型的基本流程
/// 图像数据->模型输入->模型输出->最终结果
/// </summary>
/// <typeparam name="MLIN">模型的输入</typeparam>
/// <typeparam name="MLOUT">模型的输出</typeparam>
/// <typeparam name="SRC">模型的原始输入(cv::Mat等)</typeparam>
/// <typeparam name="DST">模型的目标输出(float类型的分数等)</typeparam>
template <typename MLIN, typename MLOUT, typename SRC, typename DST>
class Model : public PopQueueData<MLIN>
{
public:
	int batchsize = IniConfig::instance().getIniInt("ModelInput", "batchsize");
public:
	//将原始输入按照batchsize大小，转换为模型输入，并存储到队列中
	void pushDataBatch(std::vector<SRC>& src);
public:
	//将图像数据转换为模型的输入
	virtual MLIN SRC2IN(std::vector<SRC>& src) = 0;
	//利用模型前向推理模型输入，得到模型输出(针对一个batchsize的数据)
	virtual MLOUT run(MLIN& in) = 0;
	//输入图像，得到模型推理图像的最终结果
	virtual std::vector<DST> run(std::vector<SRC>& src);
	//将模型输出转换为模型最终结果(针对一个batchsize的数据)
	virtual std::vector<DST> OUT2DST(MLOUT& out) = 0;

	virtual ~Model();
};

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
void Model<MLIN, MLOUT, SRC, DST>::pushDataBatch(std::vector<SRC>& src)
{
	int start = 0;
	for (int i = 0; i < src.size(); i = i + batchsize)
	{
		auto iterBegin = src.begin() + start;
		auto iterEnd = src.end();
		if (iterBegin + batchsize < iterEnd)
		{
			iterEnd = iterBegin + batchsize;
			start = i + batchsize;
		}
		std::vector<SRC> tempSrc(iterBegin, iterEnd);
		MLIN in = SRC2IN(tempSrc);
		PopQueueData<MLIN>::pushData(in);
		//std::cout << "enter data into queue" << std::endl;
	}
}

template <typename MLIN, typename MLOUT, typename SRC, typename DST>
std::vector<DST> Model<MLIN, MLOUT, SRC, DST>::run(std::vector<SRC>& src)
{
	std::vector<DST> ret;
	//模型运算基本思路
	//1.转换数据，由SRC -> MLIN
	//std::unique_lock<std::mutex> task_lock(TaskThread::task_mutex);
	auto task = std::make_shared<std::packaged_task<void()>>(
		std::bind(&Model::pushDataBatch, this, std::ref(src)));
	std::queue<TaskThread::Task> tasks;
	int modelThread = IniConfig::instance().getIniInt("Thread", "Model");
	//for(int i )
	tasks.emplace(
		[task]() {
			(*task)();
		}
	);
	TaskThread::enterTask(tasks, modelThread);

	int loopTime = std::ceil(float(src.size()) / float(batchsize));
	for (int i = 0; i < loopTime; i++)
	{
		//2.运行模型，由MLIN->MLOUT
		MLIN in;
		//std::cout << "before popdata" << std::endl;
		if (PopQueueData<MLIN>::popData(in))
		{
			//std::cout << "end popdata" << std::endl;
			MLOUT out = run(in);
			//3.转换数据，由MLOUT->DST
			std::vector<DST> dst = OUT2DST(out);
			ret.insert(ret.end(), dst.begin(), dst.end());
			//std::cout << ret.size() << std::endl;
		}
		else
		{
			std::cout << "pop error" << std::endl;
		}
	}
	return ret;
}

template<typename MLIN, typename MLOUT, typename SRC, typename DST>
Model<MLIN, MLOUT, SRC, DST>::~Model()
{

}

#endif