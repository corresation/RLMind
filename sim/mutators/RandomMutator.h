#pragma once
#include "StateMutator.h"

namespace RLGC {
	class RandomMutator : public StateMutator {
	public:
		bool
			randBallSpeed, randCarSpeed, carsOnGround;

		RandomMutator(bool randBallSpeed, bool randCarSpeed, bool carsOnGround) :
			randBallSpeed(randBallSpeed), randCarSpeed(randCarSpeed), carsOnGround(carsOnGround) {
		}

		virtual void ResetArena(Arena* arena);
	};
}