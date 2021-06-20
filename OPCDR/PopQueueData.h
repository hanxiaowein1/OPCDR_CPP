#pragma once

#ifndef _AILAB_POPQUEUEDATA_H_
#define _AILAB_POPQUEUEDATA_H_

#include "TaskThread.h"

/// <summary>
/// 多线程安全队列
/// </summary>
/// <typeparam name="T">队列中数据格式</typeparam>
template <typename T>
class PopQueueData : public TaskThread
{
public:
	//数据队列	
	std::queue<T> data_queue;
	//数据队列互斥量
	std::mutex data_mutex;
	//数据队列条件变量
	std::condition_variable data_cv;
public:
	void popQueueWithoutLock(std::vector<T>& pop_datas);
	void popQueueWithoutLock(T& pop_data);
	//弹出数据队列中所有的数据
	bool popData(std::vector<T>& pop_datas);
	//弹出数据队列的单个数据
	bool popData(T& pop_data);
	//进队
	void pushData(std::vector<T>& push_datas);
	//进队
	void pushData(T& push_data);
	virtual ~PopQueueData();
};


template <typename T>
void PopQueueData<T>::popQueueWithoutLock(T& pop_data)
{
	if (!data_queue.empty())
	{
		pop_data = std::move(data_queue.front());
		data_queue.pop();
	}
}

template <typename T>
void PopQueueData<T>::popQueueWithoutLock(std::vector<T>& pop_datas)
{
	int size = data_queue.size();
	for (int i = 0; i < size; i++)
	{
		pop_datas.emplace_back(std::move(data_queue.front()));
		data_queue.pop();
	}
}

template <typename T>
void PopQueueData<T>::pushData(std::vector<T>& push_datas)
{
	std::unique_lock<std::mutex> data_lock(data_mutex);
	for (auto iter = push_datas.begin(); iter != push_datas.end(); iter++)
	{
		data_queue.emplace(std::move(*iter));
	}
	data_lock.unlock();
	data_cv.notify_one();
}

template <typename T>
void PopQueueData<T>::pushData(T& push_data)
{
	std::unique_lock<std::mutex> data_lock(data_mutex);
	data_queue.emplace(std::move(push_data));
	data_lock.unlock();
	data_cv.notify_one();
}

template<typename T>
inline PopQueueData<T>::~PopQueueData()
{
}

template <typename T>
bool PopQueueData<T>::popData(T& pop_data)
{
	//std::cout << "------------------popData--------------------------" << std::endl;
	std::unique_lock<std::mutex> data_lock(data_mutex);
	if (data_queue.size() > 0)
	{
		popQueueWithoutLock(pop_data);
		data_lock.unlock();
		//std::cout << "111" << std::endl;
		return true;
	}
	else
	{
		data_lock.unlock();
		//取得tasks的锁，检查是否还有任务
		std::unique_lock<std::mutex> task_lock(task_mutex);
		if (multi_tasks.find(key) != multi_tasks.end())
		{
			//std::cout << "popData key:" << key;
			//std::cout << " , running thread num" << multi_tasks[key].runningThreadNum;
			//std::cout << ", tasks num" << multi_tasks[key].tasks.size() << std::endl;
			if (multi_tasks[key].runningThreadNum == 0)
			{
				if (multi_tasks[key].tasks.empty())
					return false;
			}
			task_lock.unlock();
			//证明有task，那么data_mutex再次锁上，因为一定会有数据唤醒它
			data_lock.lock();
			using namespace std;
			while (!data_cv.wait_for(data_lock, 1000ms, [this] {
				if (data_queue.size() > 0 || stopped.load() || multi_tasks.find(key) == multi_tasks.end()) {
					return true;
				}
				return false;
				})) {
			}
			//data_cv.wait_for(data_lock, 1000ms, [this] {
			//	if (data_queue.size() > 0 || stopped.load() || multi_tasks.find(key) == multi_tasks.end()) {
			//		return true;
			//	}
			//	return false;
			//	});
			if (stopped.load() || multi_tasks.find(key) == multi_tasks.end())
				return false;
			//data_cv.wait(data_lock, [this] {
			//	if (data_queue.size() > 0 || stopped.load()) {
			//		return true;
			//	}
			//	else {
			//		return false;
			//	}
			//	});

			//std::cout << "1.555555" << std::endl;
			popQueueWithoutLock(pop_data);
			//std::cout << "222" << std::endl;
			data_lock.unlock();
			return true;
		}
		//或许在findkey的时候任务刚刚跑完，然后又恰好进入了队列，这个时候再做最后一次判断
		data_lock.lock();
		if (data_queue.size() > 0)
		{
			popQueueWithoutLock(pop_data);
			//std::cout << "333" << std::endl;
			data_lock.unlock();
			return true;
		}
		//std::cout << "error" << std::endl;
		return false;
	}
}

template <typename T>
bool PopQueueData<T>::popData(std::vector<T>& pop_datas)
{
	std::unique_lock<std::mutex> data_lock(data_mutex);
	if (data_queue.size() > 0)
	{
		popQueueWithoutLock(pop_datas);
		data_lock.unlock();
		return true;
	}
	else
	{
		data_lock.unlock();
		//取得tasks的锁，检查是否还有任务
		std::unique_lock<std::mutex> task_lock(task_mutex);
		if (TaskThread::multi_tasks.find(key) != TaskThread::multi_tasks.end())
		{
			//std::cout << "popData key:" << key;
			//std::cout << " , running thread num" << TaskThread::multi_tasks[key].runningThreadNum;
			//std::cout << ", tasks num" << TaskThread::multi_tasks[key].tasks.size() << std::endl;
			if (TaskThread::multi_tasks[key].runningThreadNum == 0)
			{
				if (multi_tasks[key].tasks.empty())
					return false;
			}
			task_lock.unlock();
			//证明有task，那么data_mutex再次锁上，因为一定会有数据唤醒它(当多线程Pop的时候，其他的一个线程会被 唤醒，但是这个可能就不被唤醒然后造成停止)
			data_lock.lock();
			using namespace std;
			while (!data_cv.wait_for(data_lock, 1000ms, [this] {
				if (data_queue.size() > 0|| TaskThread::stopped.load()|| TaskThread::multi_tasks.find(key) == TaskThread::multi_tasks.end()) {
					return true;
				}
				return false;
				})) {
				continue;
			}
			//data_cv.wait_for(data_lock, 1000ms, [this] {
			//	if (data_queue.size() > 0) {
			//		std::cout << "data queue size bigger than 0" << std::endl;
			//		return true;
			//	}
			//	if (TaskThread::stopped.load()) {
			//		std::cout << "stopped load" << std::endl;
			//		return true;
			//	}
			//	if (TaskThread::multi_tasks.find(key) == TaskThread::multi_tasks.end()) {
			//		std::cout << "multi tasks key not find" << std::endl;
			//		return true;
			//	}
			//	std::cout << "data cv still wait" << std::endl;
			//	return false;
			//	});
			//data_cv.wait(data_lock, [this] {
			//	if (data_queue.size() > 0 || stopped.load()) {
			//		return true;
			//	}
			//	return false;
			//	});
			if (stopped.load() || TaskThread::multi_tasks.find(key) == TaskThread::multi_tasks.end())
				return false;
			popQueueWithoutLock(pop_datas);
			data_lock.unlock();
			return true;
		}
		//或许在findkey的时候任务刚刚跑完，然后又恰好进入了队列，这个时候再做最后一次判断
		data_lock.lock();
		if (data_queue.size() > 0)
		{
			popQueueWithoutLock(pop_datas);
			data_lock.unlock();
			return true;
		}
		return false;
	}
}


#endif