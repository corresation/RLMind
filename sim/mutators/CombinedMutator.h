#pragma once

#include "StateMutator.h"

#include "../util/Math.h"

namespace RLGC {
	// Combine state mutators with weights
	// On env reset, one of the child state mutators will be chosen at random, weighted by their corresponding weight
	class CombinedMutator : public StateMutator {
	private:
		std::vector<StateMutator*> setters = {};
		std::vector<float> cumulativeWeights = {};
		float totalWeight;

	public:
		CombinedMutator(const std::vector<std::pair<StateMutator*, float>>& setters) {

			RG_ASSERT(!setters.empty());

			totalWeight = 0;
			for (auto& pair : setters) {
				this->setters.push_back(pair.first);

				if (pair.second < 0 || isnan(pair.second) || isinf(pair.second)) {
					RG_ERR_CLOSE("CombinedMutator: Mutator \"" << typeid(*pair.first).name() << "\" has a negative or invalid weight of " << pair.second);
				}

				totalWeight += pair.second;
				cumulativeWeights.push_back(totalWeight);
			}
		}

		void ResetArena(Arena* arena) override {
			float f = RocketSim::Math::RandFloat(0, totalWeight);

			for (int i = 0; i < setters.size(); i++) {
				if (f <= cumulativeWeights[i]) {
					setters[i]->ResetArena(arena);
					return;
				}
			}

			RG_ERR_CLOSE("CombinedMutator ran out of setters before the matching cumulative weight was found (this should never happen)");
		}
	};
}