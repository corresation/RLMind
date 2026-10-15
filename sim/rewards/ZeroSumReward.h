#pragma once
#include "RewardWrapper.h"

namespace RLGC {
	// This is a wrapper class that makes another reward function zero-sum and team-distributed
	// Per-player reward is calculated using: ownReward*(1-teamSpirit) + avgTeamReward*teamSpirit - avgOpponentReward
	class ZeroSumReward : public RewardWrapper {
	public:

		float teamSpirit, opponentScale;

		// For reward logging
		std::vector<float> _lastRewards = {};

		// Team spirit is the fraction of reward shared between teammates
		// Opponent scale is the scale of punishment for opponent rewards (normally 1, non-1 is no longer zero-sum)
		ZeroSumReward(Reward* child, float teamSpirit, float opponentScale = 1, bool ownsFunc = true)
			: RewardWrapper(child), teamSpirit(teamSpirit), opponentScale(opponentScale) {

		}

	protected: 

		// Get all rewards for all players
		virtual std::vector<float> GetAllRewards(const GameState& state, bool final) override;
	};
}