#pragma once

#ifndef _OPCDR_RESNET50_H_
#define _OPCDR_RESNET50_H_

//using ResnetDST = float;
#include <string>
//Resnet50?????ս???
class ResnetDST {
public:
	float mScore;
	std::string mType;
public:
	ResnetDST() {}
	ResnetDST(float score) {
		mScore = score;
		if (mScore > 0.5f) {
			mType = "HSIL";
		}
		else {
			mType = "Normal";
		}
	}

	bool operator>(const ResnetDST& result)
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