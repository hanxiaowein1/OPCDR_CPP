#pragma once
#ifndef _AILAB_TASKTHREAD_H_
#define _AILAB_TASKTHREAD_H_

#include <vector>
#include <map>
#include <string>
#include <mutex>
#include <atomic>
#include <future>
#include <queue>
#include <functional>
#include "MD5.h"
#include "MyTime.h"


class TaskThread
{
public:
	//alias
	using Task = std::function<void()>;
	struct TaskInfo 
	{
		std::queue<Task> tasks;
		std::atomic<int> runningThreadNum = 0;
		std::atomic<int> maxThreadNum = 1;
	};

	//thread pool
	static std::vector<std::thread> pool;
	// task
	static std::map<std::string, TaskInfo> multi_tasks;
	static std::atomic<int> totalTaskNum;

	static std::condition_variable task_cv;
	static std::mutex task_mutex;
	static std::atomic<bool> stopped;//停止线程的标志
	static std::atomic<int> idlThrNum;//闲置线程数量
	static std::atomic<int> totalThrNum;//总共线程数量
	static bool my_once_flag;
	static int object_num;


	std::string key;
public:
	//static void initialize();
	void createThreadPool(int threadNum);
	void setThreadNum(int threadNum);
public:
	virtual void decreaseRunningThread(std::string current_key);
	void increaseRunningThread();
	void enterTask(std::queue<Task> &tasks, int thread_num = 2);
public:
	TaskThread();
	virtual ~TaskThread();
};

#endif