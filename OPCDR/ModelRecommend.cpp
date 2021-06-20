#include "ModelRecommend.h"

ModelRecommend::ModelRecommend(std::string hostName, std::string userName, std::string password) : MySqlBase(hostName, userName, password)
{
	std::cout << "Table ModelRecommend connected" << std::endl;
}

ModelRecommend::~ModelRecommend()
{
}

bool ModelRecommend::insert(ModelRecommendAttr attr)
{
	using namespace std;
	sql::PreparedStatement* pstmt;
	pstmt = con->prepareStatement("INSERT INTO model_recommend(id, mid, sid, finished) VALUES (?, ?, ?, ?)");
	pstmt->setInt(1, attr.id);
	pstmt->setString(2, attr.mid);
	pstmt->setInt(3, attr.sid);
	pstmt->setString(4, attr.finished);
	bool result = update(pstmt);
	delete pstmt;
	return result;
}

bool ModelRecommend::updateStatus(ModelRecommendAttr attr)
{
	using namespace std;
	sql::PreparedStatement* pstmt;
	pstmt = con->prepareStatement("UPDATE model_recommend set finished = ? where id = ?");
	pstmt->setString(1, attr.finished);
	pstmt->setInt(2, attr.id);
	bool result = update(pstmt);
	delete pstmt;
	return result;
}

std::vector<ModelRecommendAttr> ModelRecommend::getModelUnFinishedSlide(std::string mid)
{
	sql::ResultSet* res;
	sql::PreparedStatement* pstmt;
	pstmt = con->prepareStatement("select * from model_recommend where mid = ? and finished = ?");
	pstmt->setString(1, mid);
	pstmt->setString(2, "F");
	res = select(pstmt);
	std::vector<ModelRecommendAttr> ret = getResult(res);
	delete res;
	delete pstmt;
	return ret;
}

std::vector<ModelRecommendAttr> ModelRecommend::getResult(sql::ResultSet* res)
{
	std::vector<ModelRecommendAttr> ret;
	if (res == nullptr) {
		std::cout << "ModelRecommend::getResult: ResultSet is null!" << std::endl;
		return ret;
	}
	while (res->next()) {
		ModelRecommendAttr attr;
		attr.id = res->getInt("id");
		attr.mid = res->getString("mid");
		attr.sid = res->getInt("sid");
		attr.finished = res->getString("finished");
		ret.emplace_back(attr);
	}
	return ret;
}