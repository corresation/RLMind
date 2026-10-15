#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include "EnvSet.h"
#include "GAE.h"
#include "obs/DefaultObs.h"
#include "obs/DefaultObsPadded.h"
#include "obs/AdvancedObs.h"
#include "rewards/CommonRewards.h"
#include "rewards/ZeroSumReward.h"
#include "actions/LookupTableAction.h"
#include "mutators/KickoffMutator.h"
#include "mutators/FuzzedKickoffMutator.h"
#include "mutators/RandomMutator.h"
#include "mutators/CombinedMutator.h"
#include "terminals/NoTouchCondition.h"
#include "terminals/GoalScoreCondition.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace RLGC;

// Python owns the child reward, so don't let the wrapper delete it
struct PyZeroSumReward : ZeroSumReward {
	using ZeroSumReward::ZeroSumReward;
	~PyZeroSumReward() { child = NULL; }
};

// Holds on to the python env objects so the rewards etc. live as long as the env set
struct PyEnvSet : EnvSet {
	std::vector<nb::object> envs;
	using EnvSet::EnvSet;
};

template <typename T>
using Array = nb::ndarray<nb::numpy, T>;

template <typename T>
Array<T> View(std::vector<T>& vec, std::initializer_list<size_t> shape) {
	return Array<T>(vec.data(), shape.size(), shape.begin(), nb::handle());
}

// Moves the vector into a numpy array that owns it
template <typename T>
Array<T> ToArray(std::vector<T>&& vec, std::initializer_list<size_t> shape) {
	auto* owned = new std::vector<T>(std::move(vec));
	nb::capsule owner(owned, [](void* p) noexcept { delete (std::vector<T>*)p; });
	return Array<T>(owned->data(), shape.size(), shape.begin(), owner);
}

// Flags are stored as bytes, numpy reads them as bools
Array<bool> ToBoolArray(std::vector<uint8_t>&& vec, std::initializer_list<size_t> shape) {
	auto* owned = new std::vector<uint8_t>(std::move(vec));
	nb::capsule owner(owned, [](void* p) noexcept { delete (std::vector<uint8_t>*)p; });
	return Array<bool>((bool*)owned->data(), shape.size(), shape.begin(), owner);
}

// Every player (or every game) at once, "game" is each player's game index
nb::dict GetStats(PyEnvSet& e) {
	size_t numPlayers = e.state.numPlayers, numGames = e.arenas.size();
	std::vector<int> game(numPlayers);
	std::vector<float> pos(numPlayers * 3), vel(numPlayers * 3), boost(numPlayers);
	std::vector<uint8_t> team(numPlayers), onGround(numPlayers), touched(numPlayers), demoed(numPlayers);
	std::vector<uint8_t> goal(numPlayers), assist(numPlayers), shot(numPlayers), save(numPlayers), bump(numPlayers), demo(numPlayers);
	std::vector<uint8_t> goalScored(numGames), padActive(numGames * CommonValues::BOOST_LOCATIONS_AMOUNT);
	std::vector<float> ballPos(numGames * 3), ballVel(numGames * 3);

	for (size_t i = 0; i < numGames; i++) {
		auto& gs = e.state.gameStates[i];
		goalScored[i] = gs.goalScored;
		for (int k = 0; k < CommonValues::BOOST_LOCATIONS_AMOUNT; k++)
			padActive[i * CommonValues::BOOST_LOCATIONS_AMOUNT + k] = gs.boostPads[k];
		for (int k = 0; k < 3; k++) {
			ballPos[i * 3 + k] = gs.ball.pos[k];
			ballVel[i * 3 + k] = gs.ball.vel[k];
		}

		size_t start = e.state.arenaPlayerStartIdx[i];
		for (size_t j = 0; j < gs.players.size(); j++) {
			auto& player = gs.players[j];
			size_t p = start + j;
			game[p] = i;
			for (int k = 0; k < 3; k++) {
				pos[p * 3 + k] = player.pos[k];
				vel[p * 3 + k] = player.vel[k];
			}
			team[p] = (uint8_t)player.team;
			boost[p] = player.boost;
			onGround[p] = player.isOnGround;
			touched[p] = player.ballTouchedStep;
			demoed[p] = player.isDemoed;

			auto& events = player.eventState;
			goal[p] = events.goal;
			assist[p] = events.assist;
			shot[p] = events.shot;
			save[p] = events.save;
			bump[p] = events.bump;
			demo[p] = events.demo;
		}
	}

	nb::dict d;
	d["game"] = ToArray(std::move(game), { numPlayers });
	d["pos"] = ToArray(std::move(pos), { numPlayers, 3 });
	d["vel"] = ToArray(std::move(vel), { numPlayers, 3 });
	d["team"] = ToArray(std::move(team), { numPlayers });
	d["boost"] = ToArray(std::move(boost), { numPlayers });
	d["onGround"] = ToBoolArray(std::move(onGround), { numPlayers });
	d["touched"] = ToBoolArray(std::move(touched), { numPlayers });
	d["demoed"] = ToBoolArray(std::move(demoed), { numPlayers });
	d["goal"] = ToBoolArray(std::move(goal), { numPlayers });
	d["assist"] = ToBoolArray(std::move(assist), { numPlayers });
	d["shot"] = ToBoolArray(std::move(shot), { numPlayers });
	d["save"] = ToBoolArray(std::move(save), { numPlayers });
	d["bump"] = ToBoolArray(std::move(bump), { numPlayers });
	d["demo"] = ToBoolArray(std::move(demo), { numPlayers });
	d["ballPos"] = ToArray(std::move(ballPos), { numGames, 3 });
	d["ballVel"] = ToArray(std::move(ballVel), { numGames, 3 });
	d["goalScored"] = ToBoolArray(std::move(goalScored), { numGames });
	d["padActive"] = ToBoolArray(std::move(padActive), { numGames, (size_t)CommonValues::BOOST_LOCATIONS_AMOUNT });
	return d;
}

nb::list VecToList(const Vec& vec) {
	nb::list list;
	list.append(vec.x);
	list.append(vec.y);
	list.append(vec.z);
	return list;
}

nb::dict PhysToDict(const PhysState& phys) {
	nb::dict d;
	d["pos"] = VecToList(phys.pos);
	d["forward"] = VecToList(phys.rotMat.forward);
	d["right"] = VecToList(phys.rotMat.right);
	d["up"] = VecToList(phys.rotMat.up);
	d["vel"] = VecToList(phys.vel);
	d["ang_vel"] = VecToList(phys.angVel);
	return d;
}

// One game in RocketSimVis's format (see its networking-format.md)
nb::dict GetRenderState(PyEnvSet& e, int gameIdx) {
	if (gameIdx < 0 || gameIdx >= e.arenas.size())
		RG_ERR_CLOSE("getRenderState(): no game " << gameIdx);
	auto& gs = e.state.gameStates[gameIdx];

	nb::list cars;
	for (auto& player : gs.players) {
		nb::dict car;
		car["car_id"] = player.carId;
		car["team_num"] = (int)player.team;
		car["phys"] = PhysToDict(player);
		car["boost_amount"] = player.boost;
		car["on_ground"] = player.isOnGround;
		car["has_flip"] = player.HasFlipOrJump();
		car["is_demoed"] = player.isDemoed;
		// Not used by RocketSimVis, the RLBot bot's tests compare against them
		car["has_jumped"] = player.hasJumped;
		car["has_double_jumped"] = player.hasDoubleJumped;
		car["has_flipped"] = player.hasFlipped;
		car["air_time_since_jump"] = player.airTimeSinceJump;
		car["world_contact_z"] = player.worldContact.hasContact ? player.worldContact.contactNormal.z : 0;

		nb::dict controls;
		controls["throttle"] = player.prevAction.throttle;
		controls["steer"] = player.prevAction.steer;
		controls["pitch"] = player.prevAction.pitch;
		controls["yaw"] = player.prevAction.yaw;
		controls["roll"] = player.prevAction.roll;
		controls["boost"] = player.prevAction.boost;
		controls["jump"] = player.prevAction.jump;
		controls["handbrake"] = player.prevAction.handbrake;
		car["controls"] = controls;
		cars.append(car);
	}

	nb::list pads, padTimers;
	for (bool active : gs.boostPads)
		pads.append(active);
	for (float timer : gs.boostPadTimers)
		padTimers.append(timer);

	nb::dict d;
	d["gamemode"] = gs.lastArena ? GAMEMODE_STRS[(int)gs.lastArena->gameMode] : "soccar";
	d["ball_phys"] = PhysToDict(gs.ball);
	d["cars"] = cars;
	d["boost_pad_states"] = pads;
	d["boost_pad_timers"] = padTimers;
	return d;
}

// Last unweighted reward of each reward function, from one random player per game
Array<float> GetLastRewards(PyEnvSet& e) {
	size_t numRewards = e.rewards[0].size();
	std::vector<float> result(e.arenas.size() * numRewards, 0);
	for (size_t i = 0; i < e.arenas.size(); i++) {
		auto& last = e.state.lastRewards[i];
		if (last.size() != numRewards && !last.empty())
			RG_ERR_CLOSE("lastRewards: every game needs the same rewards");
		std::copy(last.begin(), last.end(), result.begin() + i * numRewards);
	}
	return ToArray(std::move(result), { e.arenas.size(), numRewards });
}

template <typename T>
T* GetObj(nb::handle env, const char* name) {
	nb::object obj = env.attr(name);
	if (obj.is_none())
		RG_ERR_CLOSE("Env." << name << " was not set");
	return nb::cast<T*>(obj);
}

PyEnvSet* MakeEnvSet(nb::list envList, int tickSkip, int actionDelay, bool saveRewards) {
	std::vector<nb::object> envs;
	std::vector<EnvCreateResult> results;
	std::unordered_set<void*> used; // Each game needs its own objects, they keep state

	auto fnUse = [&](void* ptr) {
		if (!used.insert(ptr).second)
			RG_ERR_CLOSE("Two games share a reward/obs builder/mutator etc, each game needs its own (make them inside makeEnv)");
	};

	int numGames = envList.size();
	for (nb::handle envHandle : envList) {
		nb::object env = nb::borrow(envHandle);

		EnvCreateResult result = {};
		result.arena = Arena::Create(nb::cast<GameMode>(env.attr("gameMode")));
		int playersPerTeam = nb::cast<int>(env.attr("playersPerTeam"));
		for (int j = 0; j < playersPerTeam; j++) {
			result.arena->AddCar(Team::BLUE);
			result.arena->AddCar(Team::ORANGE);
		}

		for (nb::handle item : env.attr("rewards")) {
			auto pair = nb::cast<std::pair<Reward*, float>>(item);
			fnUse(pair.first);
			result.rewards.push_back(WeightedReward(pair.first, pair.second));
		}
		for (nb::handle item : env.attr("terminals")) {
			result.terminalConditions.push_back(nb::cast<TerminalCondition*>(item));
			fnUse(result.terminalConditions.back());
		}

		result.obsBuilder = GetObj<ObsBuilder>(env, "obsBuilder");
		result.actionParser = GetObj<ActionParser>(env, "actionParser");
		result.stateMutator = GetObj<StateMutator>(env, "mutator");
		fnUse(result.obsBuilder);
		fnUse(result.actionParser);
		fnUse(result.stateMutator);

		envs.push_back(env);
		results.push_back(result);
	}

	EnvSetConfig config = {};
	config.envCreateFn = [results](int idx) { return results[idx]; };
	config.numArenas = numGames;
	config.tickSkip = tickSkip;
	config.actionDelay = actionDelay;
	config.saveRewards = saveRewards;

	PyEnvSet* envSet;
	{
		nb::gil_scoped_release release;
		envSet = new PyEnvSet(config);
	}
	envSet->envs = std::move(envs);
	return envSet;
}

NB_MODULE(_sim, m) {
	m.def("init", [](std::string meshesPath, bool silent) { RocketSim::Init(meshesPath, silent); }, "meshesPath"_a = "collision_meshes", "silent"_a = false);

	nb::enum_<GameMode>(m, "GameMode")
		.value("soccar", GameMode::SOCCAR)
		.value("hoops", GameMode::HOOPS)
		.value("heatseeker", GameMode::HEATSEEKER)
		.value("snowday", GameMode::SNOWDAY);

	nb::class_<PyEnvSet>(m, "EnvSet")
		.def(nb::new_(&MakeEnvSet), "envs"_a, "tickSkip"_a = 8, "actionDelay"_a = 7, "saveRewards"_a = true)
		.def_prop_ro("tickSkip", [](PyEnvSet& e) { return e.config.tickSkip; })
		.def_prop_ro("actionDelay", [](PyEnvSet& e) { return e.config.actionDelay; })
		.def("reset", &EnvSet::Reset, nb::call_guard<nb::gil_scoped_release>())
		.def("stepFirstHalf", [](PyEnvSet& e) { e.StepFirstHalf(true); }, nb::call_guard<nb::gil_scoped_release>())
		.def("stepSecondHalf", [](PyEnvSet& e, nb::ndarray<int, nb::ndim<1>, nb::c_contig, nb::device::cpu> actions) {
			if (actions.shape(0) != e.state.numPlayers)
				RG_ERR_CLOSE("stepSecondHalf(): got " << actions.shape(0) << " actions for " << e.state.numPlayers << " players");
			IList actionList(actions.data(), actions.data() + actions.shape(0));
			nb::gil_scoped_release release;
			e.Sync();
			e.StepSecondHalf(actionList, false);
		}, "actions"_a)
		.def_prop_ro("numGames", [](PyEnvSet& e) { return e.arenas.size(); })
		.def_prop_ro("numPlayers", [](PyEnvSet& e) { return e.state.numPlayers; })
		.def_prop_ro("obsSize", [](PyEnvSet& e) { return e.state.obs.size[1]; })
		.def_prop_ro("numActions", [](PyEnvSet& e) { return e.state.actionMasks.size[1]; })
		.def_prop_ro("obs", [](PyEnvSet& e) {
			auto& obs = e.state.obs;
			return View(obs.data, { obs.size[0], obs.size[1] });
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("actionMasks", [](PyEnvSet& e) {
			auto& masks = e.state.actionMasks;
			return View(masks.data, { masks.size[0], masks.size[1] });
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("rewards", [](PyEnvSet& e) {
			return View(e.state.rewards, { e.state.rewards.size() });
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("terminals", [](PyEnvSet& e) {
			return View(e.state.terminals, { e.state.terminals.size() });
		}, nb::rv_policy::reference_internal)
		.def_prop_ro("arenaPlayerStartIdx", [](PyEnvSet& e) {
			return View(e.state.arenaPlayerStartIdx, { e.state.arenaPlayerStartIdx.size() });
		}, nb::rv_policy::reference_internal)
		.def("getStats", &GetStats)
		.def("getLastRewards", &GetLastRewards)
		.def("getRenderState", &GetRenderState, "gameIdx"_a = 0)
		.def_prop_ro("rewardNames", [](PyEnvSet& e) {
			std::vector<std::string> names;
			for (auto& weighted : e.rewards[0])
				names.push_back(weighted.reward->GetName());
			return names;
		});

	m.def("gae", [](
		nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> rews,
		nb::ndarray<int8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> terminals,
		nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> values,
		nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> truncValues,
		float gamma, float lambda, float returnStd, float clipRange) {

		size_t n = rews.shape(0);
		if (terminals.shape(0) != n || values.shape(0) != n || truncValues.shape(0) != n)
			RG_ERR_CLOSE("gae(): all inputs need the same length");

		std::vector<float> advantages(n), returns(n);
		float clipPortion = GAE::Compute(
			n, rews.data(), terminals.data(), values.data(), truncValues.data(),
			advantages.data(), returns.data(),
			gamma, lambda, returnStd, clipRange
		);
		return nb::make_tuple(ToArray(std::move(advantages), { n }), ToArray(std::move(returns), { n }), clipPortion);
	}, "rews"_a, "terminals"_a, "values"_a, "truncValues"_a, "gamma"_a, "lam"_a, "returnStd"_a, "clipRange"_a);

	// Obs builders
	nb::class_<ObsBuilder>(m, "ObsBuilder");
	nb::class_<DefaultObs, ObsBuilder>(m, "DefaultObs").def(nb::init<>());
	nb::class_<DefaultObsPadded, DefaultObs>(m, "DefaultObsPadded").def(nb::init<int>(), "maxPlayers"_a);
	nb::class_<AdvancedObs, ObsBuilder>(m, "AdvancedObs").def(nb::init<>());

	// Action parsers
	nb::class_<ActionParser>(m, "ActionParser");
	nb::class_<LookupTableAction, ActionParser>(m, "LookupTableAction").def(nb::init<>());

	// State setters
	nb::class_<StateMutator>(m, "StateMutator");
	nb::class_<KickoffMutator, StateMutator>(m, "KickoffMutator").def(nb::init<>());
	nb::class_<FuzzedKickoffMutator, StateMutator>(m, "FuzzedKickoffMutator").def(nb::init<>());
	nb::class_<RandomMutator, StateMutator>(m, "RandomMutator")
		.def(nb::init<bool, bool, bool>(), "randBallSpeed"_a, "randCarSpeed"_a, "carsOnGround"_a);
	nb::class_<CombinedMutator, StateMutator>(m, "CombinedMutator")
		.def(nb::init<const std::vector<std::pair<StateMutator*, float>>&>(), "mutators"_a, nb::keep_alive<1, 2>());

	// Terminal conditions
	nb::class_<TerminalCondition>(m, "TerminalCondition");
	nb::class_<NoTouchCondition, TerminalCondition>(m, "NoTouchCondition").def(nb::init<float>(), "maxTime"_a);
	nb::class_<GoalScoreCondition, TerminalCondition>(m, "GoalScoreCondition").def(nb::init<>());

	// Rewards
	nb::class_<Reward>(m, "Reward");
#define REWARD(name) nb::class_<name, Reward>(m, #name).def(nb::init<>())
	REWARD(PlayerGoalReward);
	REWARD(AssistReward);
	REWARD(ShotReward);
	REWARD(ShotPassReward);
	REWARD(SaveReward);
	REWARD(BumpReward);
	REWARD(BumpedPenalty);
	REWARD(DemoReward);
	REWARD(DemoedPenalty);
	REWARD(VelocityPlayerToBallReward);
	REWARD(FaceBallReward);
	REWARD(TouchBallReward);
	REWARD(SpeedReward);
	REWARD(WavedashReward);
	REWARD(PickupBoostReward);
	REWARD(AirReward);
	REWARD(TouchAccelReward);
#undef REWARD
	nb::class_<GoalReward, Reward>(m, "GoalReward").def(nb::init<float>(), "concedeScale"_a = -1);
	nb::class_<VelocityReward, Reward>(m, "VelocityReward").def(nb::init<bool>(), "isNegative"_a = false);
	nb::class_<VelocityBallToGoalReward, Reward>(m, "VelocityBallToGoalReward").def(nb::init<bool>(), "ownGoal"_a = false);
	nb::class_<SaveBoostReward, Reward>(m, "SaveBoostReward").def(nb::init<float>(), "exponent"_a = 0.5f);
	nb::class_<StrongTouchReward, Reward>(m, "StrongTouchReward")
		.def(nb::init<float, float>(), "minSpeedKPH"_a = 20, "maxSpeedKPH"_a = 130);
	nb::class_<PyZeroSumReward, Reward>(m, "ZeroSumReward")
		.def(nb::init<Reward*, float, float>(), "child"_a, "teamSpirit"_a, "opponentScale"_a = 1, nb::keep_alive<1, 2>());
}
