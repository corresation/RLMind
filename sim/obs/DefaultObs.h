#pragma once
#include "ObsBuilder.h"

// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/obs_builders/default_obs.py
namespace RLGC {
	class DefaultObs : public ObsBuilder {
	public:

		Vec posCoef;
		float velCoef, angVelCoef;
		DefaultObs(
			Vec posCoef = Vec(1 / CommonValues::SIDE_WALL_X, 1 / CommonValues::BACK_WALL_Y, 1 / CommonValues::CEILING_Z),
			float velCoef = 1 / CommonValues::CAR_MAX_SPEED,
			float angVelCoef = 1 / CommonValues::CAR_MAX_ANG_VEL
		) : posCoef(posCoef), velCoef(velCoef), angVelCoef(angVelCoef) {

		}

		virtual void AddPlayerToObs(FList& obs, const Player& player, bool inv);

		virtual FList BuildObs(const Player& player, const GameState& state);
	};
}