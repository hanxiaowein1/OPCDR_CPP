#pragma once
#ifndef _OPCDR_MOBILENET_H_
#define _OPCDR_MOBILENET_H_

//using MobileNetDST = float;

#include <string>
//mobilenet的输出结果类
class MobileNetDST {
public:
	float mScore;
	std::string mType;
public:
	MobileNetDST() {}
	MobileNetDST(float score) {
		mScore = score;
		if (mScore > 0.5f) {
			mType = "HSIL";
		}
		else {
			mType = "Normal";
		}
	}

	bool operator>(const MobileNetDST& result)
	{
		if (mScore > result.mScore)
			return true;
		return false;
	}

	std::string getType() {
		return mType;
	}

	float getScore() {
		return mScore;
	}
};

#endif