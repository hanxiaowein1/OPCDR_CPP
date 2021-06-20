#include "Slides.h"

Slides::Slides(std::string hostName, std::string userName, std::string password) :
	MySqlBase(hostName, userName, password) 
{
	std::cout << "Table Slides connected" << std::endl;
}

Slides::~Slides()
{
}

SlidesAttr Slides::getSlideBySid(int sid)
{
	sql::PreparedStatement* pstmt = nullptr;
	pstmt = con->prepareStatement("select * from slides where sid = ?");
	pstmt->setInt(1, sid);
	auto res = select(pstmt);
	SlidesAttr ret;
	std::vector<SlidesAttr> temp_result;
	//std::string ret = "";
	temp_result = getResult(res);
	if (temp_result.size() > 0) {
		ret = temp_result[0];
	}
	delete res;
	delete pstmt;
	return ret;
}

std::string Slides::getSlideNameBySid(int sid)
{
	//std::string sql = "select slide_name from slides where sid"
	sql::PreparedStatement* pstmt = nullptr;
	pstmt = con->prepareStatement("select slide_name from slides where sid = ?");
	pstmt->setInt(1, sid);
	auto res = select(pstmt);
	std::string ret = "";
	while (res->next()) {
		ret = res->getString("slide_name");
	}
	delete res;
	delete pstmt;
	return ret;
}

std::vector<SlidesAttr> Slides::getResult(sql::ResultSet* res)
{
	std::vector<SlidesAttr> ret;
	if (res == nullptr) {
		std::cout << "ModelRecommend::getResult: ResultSet is null!" << std::endl;
		return ret;
	}
	while (res->next()) {
		SlidesAttr attr;
		attr.sid = res->getInt("sid");
		attr.slide_path = res->getString("slide_path");
		attr.slide_name = res->getString("slide_name");
		attr.slide_group = res->getString("slide_group");
		attr.pro_method = res->getString("pro_method");
		attr.image_method = res->getString("image_method");
		attr.mpp = res->getDouble("mpp");
		attr.zoom = res->getString("zoom");
		attr.slide_format = res->getString("slide_format");
		attr.is_positive = res->getString("is_positive");
		attr.width = res->getInt("width");
		attr.height = res->getInt("height");
		attr.bounds_x = res->getInt("bounds_x");
		attr.bounds_y = res->getInt("bounds_y");
		attr.format_trans = res->getString("format_trans");
		ret.emplace_back(attr);
	}
	return ret;                    
}