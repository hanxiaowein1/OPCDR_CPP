#pragma once
#ifndef _OPDCR_ANNORECOMMEND_H_
#define _OPDCR_ANNORECOMMEND_H_

#include "MySqlBase.h"

struct AnnoRecommendAttr
{
	int id;
	int sid;
	std::string mid;
	int aid;
	std::string cir_rect;
	std::string anno_class;
	std::string center_point;
	std::string top_left;
	std::string type = "Rect";
};

class AnnoRecommend : MySqlBase<AnnoRecommendAttr>
{
public:
	AnnoRecommend() = delete;
	AnnoRecommend(std::string hostName, std::string userName, std::string password);
	~AnnoRecommend();
	virtual std::vector<AnnoRecommendAttr> getResult(sql::ResultSet* res);
	bool insert(AnnoRecommendAttr attr);
	int getMaxId();
};

#endif