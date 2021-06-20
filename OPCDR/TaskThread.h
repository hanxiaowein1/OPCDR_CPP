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
	//任务队列类
	struct TaskInfo 
	{
		//任务队列
		std::queue<Task> tasks;
		//任务队列中正在执行的数量
		std::atomic<int> runningThreadNum = 0;
		//任务队列最大的可使用线程数量
		std::atomic<int> maxThreadNum = 1;
	};

	//thread pool
	static std::vector<std::thread> pool;
	// 多任务队列，key为string，value为任务队列类对象
	static std::map<std::string, TaskInfo> multi_tasks;
	//所有任务队列的任务个数
	static std::atomic<int> totalTaskNum;

	//任务的条件变量
	static std::condition_variable task_cv;
	//任务的锁
	static std::mutex task_mutex;
	//停止线程的标志
	static std::atomic<bool> stopped;
	//闲置线程数量
	static std::atomic<int> idlThrNum;
	//总共线程数量
	static std::atomic<int> totalThrNum;
	//确保函数只执行一次
	static bool my_once_flag;
	//TaskThread的对象个数
	static int object_num;

	//任务队列的key
	std::string key;
public:
	//static void initialize();
	//创建线程池
	void createThreadPool(int threadNum);
	//为任务队列设置最大可使用线程数
	void setThreadNum(int threadNum);
public:
	//任务队列正在执行的线程数减一
	virtual void decreaseRunningThread(std::string current_key);
	//任务队列正在执行的线程数加一
	void increaseRunningThread();
	//为任务队列添加任务，并添加最大可执行线程数
	void enterTask(std::queue<Task> &tasks, int thread_num = 2);
public:
	TaskThread();
	virtual ~TaskThread();
};

#endif