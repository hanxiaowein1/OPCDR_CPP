#pragma once
#ifndef _OPCDR_SLIDES_H_
#define _OPCDR_SLIDES_H_

#include "MySqlBase.h"

struct SlidesAttr
{
	int sid;
	std::string slide_path;
	std::string slide_name;
	std::string slide_group;
	std::string pro_method;
	std::string image_method;
	float mpp;
	std::string zoom;
	std::string slide_format;
	std::string is_positive;
	int width;
	int height;
	int bounds_x;
	int bounds_y;
	std::string format_trans;
};

class Slides : public MySqlBase<SlidesAttr>
{
public:
	Slides(std::string hostName, std::string userName, std::string password);
	~Slides();

	virtual std::vector<SlidesAttr> getResult(sql::ResultSet* res);
	std::string getSlideNameBySid(int sid);
	SlidesAttr getSlideBySid(int sid);
};


#endif
