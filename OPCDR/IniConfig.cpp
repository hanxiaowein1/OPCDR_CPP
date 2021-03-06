#include "IniConfig.h"
#include <windows.h>

//const char* const IniConfig::m_inipath = "config.ini";
std::string m_inipath = "config.ini";

void setIniPath(std::string inipath)
{
	m_inipath = inipath;
}

IniConfig::IniConfig()
{

}

IniConfig::~IniConfig()
{

}

IniConfig& IniConfig::instance()
{
	// TODO: ?ڴ˴????? return ????
	static IniConfig instance;
	return instance;
}

std::string IniConfig::getIniString(std::string group, std::string key)
{
	std::string ret;
	char buffer[MAX_PATH];
	GetPrivateProfileString(group.c_str(), key.c_str(), "default", buffer, MAX_PATH, m_inipath.c_str());
	ret = std::string(buffer);
	return ret;
}

void IniConfig::setIniString(std::string group, std::string key, std::string value)
{
	WritePrivateProfileString(group.c_str(), key.c_str(), value.c_str(), m_inipath.c_str());
}

int IniConfig::getIniInt(std::string group, std::string key)
{
	int ret = GetPrivateProfileInt(group.c_str(), key.c_str(), -1, m_inipath.c_str());
	return ret;
}

void IniConfig::setIniInt(std::string group, std::string key, int value)
{
	setIniString(group, key, std::to_string(value));
}

double IniConfig::getIniDouble(std::string group, std::string key)
{
	double ret;
	std::string ret_str = getIniString(group, key);
	if (ret_str == "default")
		return -1;
	else
	{
		ret = std::stod(ret_str);
		return ret;
	}
}

void IniConfig::setIniDouble(std::string group, std::string key, double value)
{
	setIniString(group, key, std::to_string(value));
}
