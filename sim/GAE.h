#pragma once
#include "util/Framework.h"

namespace RLGC {
	namespace GAE {
		// Rows are whole episodes back to back, each ending in a terminal
		// truncValues holds the next state's value on truncated rows
		// Returns the portion of reward that got clipped
		float Compute(
			int numRows, const float* rews, const int8_t* terminals, const float* values, const float* truncValues,
			float* outAdvantages, float* outReturns,
			float gamma, float lambda, float returnStd, float clipRange
		);
	}
}
