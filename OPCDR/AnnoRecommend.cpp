#include "AnnoRecommend.h"

AnnoRecommend::AnnoRecommend(std::string hostName, std::string userName, std::string password) : 
	MySqlBase(hostName, userName, password)
{
}

AnnoRecommend::~AnnoRecommend()
{
}

bool AnnoRecommend::insert(AnnoRecommendAttr attr)
{
	sql::PreparedStatement* pstmt;
	pstmt = con->prepareStatement(
		"INSERT INTO anno_recommend(sid, mid, anno_class, center_point, top_left, type) VALUES (?, ?, ?, ?, ?, ?)");
	//pstmt->setInt(1, attr.id);
	pstmt->setInt(1, attr.sid);
	pstmt->setString(2, attr.mid);
	pstmt->setString(3, attr.anno_class);
	pstmt->setString(4, attr.center_point);
	pstmt->setString(5, attr.top_left);
	pstmt->setString(6, attr.type);
	bool result = update(pstmt);
	delete pstmt;
	return result;
}

int AnnoRecommend::getMaxId()
{
	int ret = -1;
	std::string sql = "select max(id) from anno_recommend";
	sql::ResultSet* res = select(sql);
	if (res) {
		while (res->next()) {
			ret = res->getInt("max(id)");
		}
	}
	delete res;
	return ret;
}

std::vector<AnnoRecommendAttr> AnnoRecommend::getResult(sql::ResultSet* res)
{
	std::vector<AnnoRecommendAttr> ret;
	if (res == nullptr) {
		std::cout << "ModelRecommend::getResult: ResultSet is null!" << std::endl;
		return ret;
	}
	while (res->next()) {
		AnnoRecommendAttr attr;
		attr.id = res->getInt("id");
		attr.sid = res->getInt("sid");
		attr.mid = res->getString("mid");
		attr.aid = res->getInt("aid");		
		attr.cir_rect = res->getString("cir_rect");
		attr.anno_class = res->getString("anno_class");
		attr.center_point = res->getString("center_point");
		attr.top_left = res->getString("top_left");
		attr.type = res->getString("type");
		ret.emplace_back(attr);
	}
	return ret;
}