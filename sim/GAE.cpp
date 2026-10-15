#include "GAE.h"
#include "terminals/TerminalCondition.h"

float RLGC::GAE::Compute(
	int numRows, const float* rews, const int8_t* terminals, const float* values, const float* truncValues,
	float* outAdvantages, float* outReturns,
	float gamma, float lambda, float returnStd, float clipRange) {

	if (numRows > 0 && terminals[numRows - 1] == TerminalType::NOT_TERMINAL)
		RG_ERR_CLOSE("GAE: last row isn't terminal, episodes must be complete");

	float prevLambda = 0, prevRet = 0;
	float totalRew = 0, totalClippedRew = 0;

	for (int step = numRows - 1; step >= 0; step--) {
		uint8_t terminal = terminals[step];
		float done = terminal == TerminalType::NORMAL;
		float trunc = terminal == TerminalType::TRUNCATED;

		float curReward;
		if (returnStd != 0) {
			curReward = rews[step] / returnStd;
			totalRew += abs(curReward);

			// We only clip if returns are standardized
			if (clipRange > 0)
				curReward = RS_CLAMP(curReward, -clipRange, clipRange);

			totalClippedRew += abs(curReward);
		} else {
			curReward = rews[step];
			totalRew += abs(curReward);
		}

		float nextValPred;
		if (trunc) {
			nextValPred = truncValues[step];
		} else if (done) {
			nextValPred = 0;
		} else {
			nextValPred = values[step + 1];
		}

		float predReturn = curReward + gamma * nextValPred * (1 - done);
		float delta = predReturn - values[step];
		float curReturn = rews[step] + prevRet * gamma * (1 - done) * (1 - trunc);
		outReturns[step] = curReturn;

		prevLambda = delta + gamma * lambda * (1 - done) * (1 - trunc) * prevLambda;
		outAdvantages[step] = prevLambda;

		prevRet = curReturn;
	}

	return (totalRew - totalClippedRew) / RS_MAX(totalRew, 1e-7f);
}
