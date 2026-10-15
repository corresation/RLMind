#pragma once
#include "MathTypes.h"

RS_NS_START

namespace Math {
	// NOTE: min is inclusive, max is exclusive
	// Seed will be used if not -1
	RSAPI int RandInt(int min, int max, int seed = -1);

	RSAPI float RandFloat(float min = 0, float max = 1);

	RSAPI std::default_random_engine& GetRandEngine();

	RSAPI float WrapNormalizeFloat(float val, float minmax);

	// Simulates aggressive UE3 rotator rounding when converting from a UE3 rotator to vector
	RSAPI Angle RoundAngleUE3(Angle ang);
}

RS_NS_END