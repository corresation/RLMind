#pragma once
#include "StateMutator.h"

namespace RLGC {
	class KickoffMutator : public StateMutator {
	public:
		void ResetArena(Arena* arena) {
			arena->ResetToRandomKickoff();
		}
	};
}