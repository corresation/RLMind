#include "AttentionObs.h"
#include "../state/StateUtil.h"

void RLGC::AttentionObs::AddTokenToObs(FList& obs, TokenType type, const PhysState& phys, const PhysState& self, const PhysState& ball, const Player* player) {
	for (int i = 0; i < TOKEN_TYPE_AMOUNT; i++)
		obs += (float)(i == type);

	obs += phys.pos * POS_COEF;
	obs += self.rotMat.Dot(phys.pos - self.pos) * POS_COEF;
	obs += phys.vel * VEL_COEF;
	obs += self.rotMat.Dot(phys.vel - self.vel) * VEL_COEF;
	obs += phys.angVel * ANG_VEL_COEF;

	if (player) {
		obs += phys.rotMat.forward;
		obs += phys.rotMat.up;

		// Local ball pos and vel
		obs += phys.rotMat.Dot(ball.pos - phys.pos) * POS_COEF;
		obs += phys.rotMat.Dot(ball.vel - phys.vel) * VEL_COEF;

		obs += player->boost / 100;
		obs += player->isOnGround;
		obs += player->HasFlipOrJump();
		obs += player->isDemoed;
		obs += player->hasJumped;
	}
	else {
		obs += FList(12 + 5);
	}
}

RLGC::FList RLGC::AttentionObs::BuildObs(const Player& player, const GameState& state) {
	FList obs = {};

	bool inv = player.team == Team::ORANGE;

	auto ball = InvertPhys(state.ball, inv);
	auto self = InvertPhys(player, inv);
	auto& pads = state.GetBoostPads(inv);
	auto& padTimers = state.GetBoostPadTimers(inv);

	for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++)
		obs += player.prevAction[i];

	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++)
		obs += pads[i] ? 1.f : 1.f / (1.f + padTimers[i]);

	AddPlayerToObs(obs, player, inv, ball);
	RG_ASSERT(obs.size() == GetQuerySize());

	AddTokenToObs(obs, BALL, ball, self, ball, NULL);

	FList teammates = {}, opponents = {};
	int numTeammates = 0, numOpponents = 0;
	for (auto& otherPlayer : state.players) {
		if (otherPlayer.carId == player.carId)
			continue;

		bool isTeammate = otherPlayer.team == player.team;
		AddTokenToObs(isTeammate ? teammates : opponents, isTeammate ? TEAMMATE : OPPONENT, InvertPhys(otherPlayer, inv), self, ball, &otherPlayer);
		(isTeammate ? numTeammates : numOpponents)++;
	}

	if (numTeammates > maxPlayers - 1)
		RG_ERR_CLOSE("AttentionObs: Too many teammates for obs, maximum is " << (maxPlayers - 1));

	if (numOpponents > maxPlayers)
		RG_ERR_CLOSE("AttentionObs: Too many opponents for obs, maximum is " << maxPlayers);

	obs += teammates;
	obs += FList((maxPlayers - 1 - numTeammates) * GetTokenSize());
	obs += opponents;
	obs += FList((maxPlayers - numOpponents) * GetTokenSize());
	return obs;
}