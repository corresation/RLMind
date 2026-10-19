#pragma once
#include "AdvancedObs.h"

namespace RLGC {
	// AdvancedObs as [query][ball][teammates][opponents], missing players are all-zero tokens
	class AttentionObs : public AdvancedObs {
	public:
		enum TokenType {
			BALL,
			TEAMMATE,
			OPPONENT,

			TOKEN_TYPE_AMOUNT
		};

		int maxPlayers;

		AttentionObs(int maxPlayers = 3) : maxPlayers(maxPlayers) {}

		void AddTokenToObs(FList& obs, TokenType type, const PhysState& phys, const PhysState& self, const PhysState& ball, const Player* player);

		virtual FList BuildObs(const Player& player, const GameState& state) override;

		// AddPlayerToObs() adds 29
		virtual int GetQuerySize() override { return Action::ELEM_AMOUNT + CommonValues::BOOST_LOCATIONS_AMOUNT + 29; }
		virtual int GetTokenSize() override { return TOKEN_TYPE_AMOUNT + 9 * 3 + 5; }
	};
}