#pragma once
#ifndef _AILAB_INICONFIG_H_
#define _AILAB_INICONFIG_H_

#include <string>

void setIniPath(std::string inipath);
//ini�����ļ��Ķ�ȡд���࣬����������Ϊȫ��Ψһ
class IniConfig
{
private:
	IniConfig();
	~IniConfig();
public:
	IniConfig& operator=(const IniConfig&) = delete;
	IniConfig& operator=(IniConfig&&) = delete;
	static IniConfig& instance();
	std::string getIniString(std::string group, std::string key);
	void setIniString(std::string group, std::string key, std::string value);
	int getIniInt(std::string group, std::string key);
	void setIniInt(std::string group, std::string key, int value);
	double getIniDouble(std::string group, std::string key);
	void setIniDouble(std::string group, std::string key, double value);
};

#endif