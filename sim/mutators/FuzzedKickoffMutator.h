#pragma once
#include "StateMutator.h"

namespace RLGC {
	// Like KickoffMutator, but very slightly randomizes the cars
	//	to prevent deterministic agents from repeating gameplay.
	class FuzzedKickoffMutator : public StateMutator {
	public:
		constexpr static float
			FUZZ_POS_RANGE = 0.1f;

		static_assert(
			(float)(3e4f + (FUZZ_POS_RANGE / 100.f)) != 3e4f,
			"FuzzedKickoffMutator::FUZZ_POS_RANGE range is too small to survive float rounding"
		);

		void ResetArena(Arena* arena) {
			arena->ResetToRandomKickoff();

			for (auto& car : arena->_cars) {
				auto state = car->GetState();
				for (int i = 0; i < 3; i++)
					state.pos[i] += RocketSim::Math::RandFloat(-FUZZ_POS_RANGE, FUZZ_POS_RANGE);
				car->SetState(state);
			}
		}
	};
}