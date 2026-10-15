#pragma once
#include "../state/GameState.h"

namespace RLGC {
	class StateMutator {
	public:
		virtual void ResetArena(Arena* arena) = 0;
	};
}