#include "Arena.h"
#include "RocketSim.h"

RS_NS_START

// These match the structs in bridge.rs
struct RsVec { float x, y, z; };
struct RsPhys { RsVec pos, forward, right, up, vel, angVel; };
struct RsControls {
	float throttle, steer, pitch, yaw, roll;
	bool jump, boost, handbrake;
};
struct RsCar {
	RsPhys phys;
	RsControls prevControls;
	bool isOnGround;
	bool wheelsWithContact[4];
	bool hasJumped, hasDoubleJumped, hasFlipped;
	RsVec flipRelTorque;
	float jumpTime, flipTime;
	bool isFlipping, isJumping;
	float airTime, airTimeSinceJump, boost, boostingTime;
	bool isSupersonic;
	float supersonicGraceTimer, handbrakeVal;
	bool isAutoFlipping;
	float autoFlipTimer, autoFlipTorqueScale;
	bool hasWorldContact;
	RsVec worldContactNormal;
	int32_t bumpVictim;
	float bumpCooldownTimer;
	bool isDemoed;
	float demoRespawnTimer;
};
struct RsPad { RsVec pos; bool isBig; float cooldown; };
struct RsMutators { RsVec gravity; float ballRadius, goalBaseThresholdY; };
enum : uint8_t { EVENT_CAR_HIT_BALL = 1, EVENT_CAR_HIT_CAR = 2 };
struct RsEvent {
	uint8_t kind;
	bool isDemo;
	uint32_t carA, carB;
	uint64_t tick;
	RsVec contactPoint, extraHitVel, ballPos;
};

extern "C" {
	bool rs_init(const char* meshesPath, bool silent);
	void* rs_arena_new(uint8_t gameMode);
	void rs_arena_free(void* arena);
	uint32_t rs_arena_add_car(void* arena, uint8_t team);
	void rs_arena_mutators(void* arena, RsMutators* out);
	uint32_t rs_arena_step(void* arena, uint32_t ticks, const RsControls* controls, RsEvent* events, uint32_t eventsCap);
	uint64_t rs_arena_tick_count(void* arena);
	uint32_t rs_arena_num_pads(void* arena);
	void rs_arena_get(void* arena, RsCar* cars, RsPhys* ball, RsPad* pads);
	void rs_arena_set_car(void* arena, uint32_t idx, const RsCar* car);
	void rs_arena_set_ball(void* arena, const RsPhys* ball);
	void rs_arena_set_pad(void* arena, uint32_t idx, float cooldown);
	void rs_arena_reset_kickoff(void* arena, int64_t seed);
	bool rs_arena_is_ball_scored(void* arena);
}

static Vec FromRs(RsVec v) { return Vec(v.x, v.y, v.z); }
static RsVec ToRs(Vec v) { return RsVec{ v.x, v.y, v.z }; }

static void FromRs(const RsPhys& from, PhysState& to) {
	to.pos = FromRs(from.pos);
	to.rotMat = RotMat(FromRs(from.forward), FromRs(from.right), FromRs(from.up));
	to.vel = FromRs(from.vel);
	to.angVel = FromRs(from.angVel);
}

static RsPhys ToRs(const PhysState& from) {
	return RsPhys{
		ToRs(from.pos),
		ToRs(from.rotMat.forward), ToRs(from.rotMat.right), ToRs(from.rotMat.up),
		ToRs(from.vel), ToRs(from.angVel)
	};
}

static CarControls FromRs(const RsControls& c) {
	CarControls result;
	result.throttle = c.throttle;
	result.steer = c.steer;
	result.pitch = c.pitch;
	result.yaw = c.yaw;
	result.roll = c.roll;
	result.jump = c.jump;
	result.boost = c.boost;
	result.handbrake = c.handbrake;
	return result;
}

static RsControls ToRs(const CarControls& c) {
	return RsControls{ c.throttle, c.steer, c.pitch, c.yaw, c.roll, c.jump, c.boost, c.handbrake };
}

// Keeps ballHitInfo and updateCounter, they're tracked on this side
static void FromRs(const RsCar& c, CarState& s) {
	FromRs(c.phys, s);
	s.lastControls = FromRs(c.prevControls);
	s.isOnGround = c.isOnGround;
	for (int i = 0; i < 4; i++)
		s.wheelsWithContact[i] = c.wheelsWithContact[i];
	s.hasJumped = c.hasJumped;
	s.hasDoubleJumped = c.hasDoubleJumped;
	s.hasFlipped = c.hasFlipped;
	s.flipRelTorque = FromRs(c.flipRelTorque);
	s.jumpTime = c.jumpTime;
	s.flipTime = c.flipTime;
	s.isFlipping = c.isFlipping;
	s.isJumping = c.isJumping;
	s.airTime = c.airTime;
	s.airTimeSinceJump = c.airTimeSinceJump;
	s.boost = c.boost;
	s.timeSpentBoosting = c.boostingTime;
	s.isSupersonic = c.isSupersonic;
	s.supersonicTime = c.supersonicGraceTimer;
	s.handbrakeVal = c.handbrakeVal;
	s.isAutoFlipping = c.isAutoFlipping;
	s.autoFlipTimer = c.autoFlipTimer;
	s.autoFlipTorqueScale = c.autoFlipTorqueScale;
	s.worldContact.hasContact = c.hasWorldContact;
	s.worldContact.contactNormal = FromRs(c.worldContactNormal);
	s.carContact.otherCarID = c.bumpVictim >= 0 ? c.bumpVictim + 1 : 0;
	s.carContact.cooldownTimer = c.bumpCooldownTimer;
	s.isDemoed = c.isDemoed;
	s.demoRespawnTimer = c.demoRespawnTimer;
}

static RsCar ToRs(const CarState& s) {
	RsCar c = {};
	c.phys = ToRs((const PhysState&)s);
	c.prevControls = ToRs(s.lastControls);
	c.isOnGround = s.isOnGround;
	for (int i = 0; i < 4; i++)
		c.wheelsWithContact[i] = s.wheelsWithContact[i];
	c.hasJumped = s.hasJumped;
	c.hasDoubleJumped = s.hasDoubleJumped;
	c.hasFlipped = s.hasFlipped;
	c.flipRelTorque = ToRs(s.flipRelTorque);
	c.jumpTime = s.jumpTime;
	c.flipTime = s.flipTime;
	c.isFlipping = s.isFlipping;
	c.isJumping = s.isJumping;
	c.airTime = s.airTime;
	c.airTimeSinceJump = s.airTimeSinceJump;
	c.boost = s.boost;
	c.boostingTime = s.timeSpentBoosting;
	c.isSupersonic = s.isSupersonic;
	c.supersonicGraceTimer = s.supersonicTime;
	c.handbrakeVal = s.handbrakeVal;
	c.isAutoFlipping = s.isAutoFlipping;
	c.autoFlipTimer = s.autoFlipTimer;
	c.autoFlipTorqueScale = s.autoFlipTorqueScale;
	c.hasWorldContact = s.worldContact.hasContact;
	c.worldContactNormal = ToRs(s.worldContact.contactNormal);
	c.bumpVictim = (int32_t)s.carContact.otherCarID - 1;
	c.bumpCooldownTimer = s.carContact.cooldownTimer;
	c.isDemoed = s.isDemoed;
	c.demoRespawnTimer = s.demoRespawnTimer;
	return c;
}

static RocketSimStage stage = RocketSimStage::UNINITIALIZED;

void Init(std::filesystem::path collisionMeshesFolder, bool silent) {
	stage = RocketSimStage::INITIALIZING;
	if (!rs_init(collisionMeshesFolder.string().c_str(), silent))
		RS_ERR_CLOSE("Failed to load collision meshes from " << collisionMeshesFolder);
	stage = RocketSimStage::INITIALIZED;
}

RocketSimStage GetStage() {
	return stage;
}

/////////////////////////////////

void Car::SetState(const CarState& state) {
	RsCar rsCar = ToRs(state);
	rs_arena_set_car(_arena->_rs, _idx, &rsCar);
	_internalState = state;
	_internalState.updateCounter = 0;
}

void Ball::SetState(const BallState& state) {
	RsPhys rsBall = ToRs((const PhysState&)state);
	rs_arena_set_ball(_arena->_rs, &rsBall);
	_internalState = state;
	_internalState.updateCounter = 0;
}

void BoostPad::SetState(const BoostPadState& state) {
	rs_arena_set_pad(_arena->_rs, _idx, state.isActive ? 0 : RS_MAX(state.cooldown, FLT_EPSILON));
	_internalState = state;
}

Arena* Arena::Create(GameMode gameMode) {
	if (stage != RocketSimStage::INITIALIZED && gameMode != GameMode::THE_VOID)
		RS_ERR_CLOSE("Arena::Create(): RocketSim::Init() must be called first");

	Arena* arena = new Arena();
	arena->gameMode = gameMode;
	arena->_rs = rs_arena_new((uint8_t)gameMode);

	RsMutators mutators;
	rs_arena_mutators(arena->_rs, &mutators);
	arena->_mutatorConfig = { FromRs(mutators.gravity), mutators.ballRadius, mutators.goalBaseThresholdY };

	arena->ball = new Ball();
	arena->ball->_arena = arena;

	int numPads = rs_arena_num_pads(arena->_rs);
	for (int i = 0; i < numPads; i++) {
		BoostPad* pad = new BoostPad();
		pad->_idx = i;
		pad->_arena = arena;
		arena->_boostPads.push_back(pad);
	}

	arena->_PullState();
	return arena;
}

Car* Arena::AddCar(Team team) {
	Car* car = new Car();
	car->team = team;
	car->_idx = rs_arena_add_car(_rs, (uint8_t)team);
	car->id = car->_idx + 1;
	car->_arena = this;
	_cars.push_back(car);
	_PullState();
	return car;
}

void Arena::SetCarBumpCallback(CarBumpEventFn callbackFn, void* userInfo) {
	_carBumpCallback.func = callbackFn;
	_carBumpCallback.userInfo = userInfo;
}

void Arena::_PullState() {
	thread_local std::vector<RsCar> cars;
	thread_local std::vector<RsPad> pads;
	cars.resize(_cars.size());
	pads.resize(_boostPads.size());
	RsPhys rsBall;
	rs_arena_get(_rs, cars.data(), &rsBall, pads.data());

	for (size_t i = 0; i < _cars.size(); i++)
		FromRs(cars[i], _cars[i]->_internalState);
	FromRs(rsBall, ball->_internalState);
	for (size_t i = 0; i < _boostPads.size(); i++) {
		auto pad = _boostPads[i];
		pad->config.pos = FromRs(pads[i].pos);
		pad->config.isBig = pads[i].isBig;
		pad->_internalState.cooldown = pads[i].cooldown;
		pad->_internalState.isActive = pads[i].cooldown <= 0;
	}
	tickCount = rs_arena_tick_count(_rs);
}

void Arena::Step(int ticksToSimulate) {
	constexpr uint32_t MAX_EVENTS = 256;
	thread_local std::vector<RsControls> controls;
	thread_local RsEvent events[MAX_EVENTS];

	controls.resize(_cars.size());
	for (size_t i = 0; i < _cars.size(); i++)
		controls[i] = ToRs(_cars[i]->controls);

	uint32_t numEvents = rs_arena_step(_rs, ticksToSimulate, controls.data(), events, MAX_EVENTS);
	if (numEvents > MAX_EVENTS)
		RS_ERR_CLOSE("Arena::Step(): too many hit events in one step (" << numEvents << "), step fewer ticks at once");

	_PullState();
	for (Car* car : _cars)
		car->_internalState.updateCounter += ticksToSimulate;
	ball->_internalState.updateCounter += ticksToSimulate;

	for (uint32_t i = 0; i < numEvents; i++) {
		auto& event = events[i];
		if (event.kind == EVENT_CAR_HIT_BALL) {
			BallHitInfo& hit = _cars[event.carA]->_internalState.ballHitInfo;
			hit.isValid = true;
			hit.ballPos = FromRs(event.ballPos);
			hit.relativePosOnBall = FromRs(event.contactPoint) - hit.ballPos;
			hit.extraHitVel = FromRs(event.extraHitVel);
			hit.tickCountWhenHit = event.tick;
		} else if (event.kind == EVENT_CAR_HIT_CAR) {
			if (_carBumpCallback.func)
				_carBumpCallback.func(this, _cars[event.carA], _cars[event.carB], event.isDemo, _carBumpCallback.userInfo);
		}
	}
}

void Arena::ResetToRandomKickoff(int seed) {
	rs_arena_reset_kickoff(_rs, seed);
	_PullState();
	for (Car* car : _cars) {
		car->_internalState.ballHitInfo = {};
		car->_internalState.updateCounter = 0;
	}
	ball->_internalState.updateCounter = 0;
}

bool Arena::IsBallScored() const {
	return rs_arena_is_ball_scored(_rs);
}

static float BallWithinHoopsGoalXYMarginSq(float x, float y) {
	constexpr float
		SCALE_Y = 0.9f,
		OFFSET_Y = 2770,
		RADIUS_SQ = 716 * 716;

	float dx = x;
	float dy = abs(y) * SCALE_Y - OFFSET_Y;
	float distSq = dx * dx + dy * dy;
	return distSq - RADIUS_SQ;
}

// Ported from RocketSim 2
bool Arena::IsBallProbablyGoingIn(float maxTime, float extraMargin, Team* goalTeamOut) const {
	Vec ballPos = ball->_internalState.pos;
	Vec ballVel = ball->_internalState.vel;
	auto& mutators = _mutatorConfig;

	if (gameMode == GameMode::SOCCAR || gameMode == GameMode::SNOWDAY) {
		if (abs(ballVel.y) < FLT_EPSILON)
			return false;

		float scoreDirSgn = RS_SGN(ballVel.y);
		float goalY = mutators.goalBaseThresholdY * scoreDirSgn;
		float distToGoal = abs(ballPos.y - goalY);

		float timeToGoal = distToGoal / abs(ballVel.y);

		if (timeToGoal > maxTime)
			return false;

		Vec extrapPosWhenScore = ballPos + (ballVel * timeToGoal) + (mutators.gravity * timeToGoal * timeToGoal) / 2;

		// From: https://github.com/RLBot/RLBot/wiki/Useful-Game-Values
		constexpr float
			APPROX_GOAL_HALF_WIDTH = 892.755f,
			APPROX_GOAL_HEIGHT = 642.775;

		float scoreMargin = mutators.ballRadius * 0.1f + extraMargin;

		if (extrapPosWhenScore.z > APPROX_GOAL_HEIGHT + scoreMargin)
			return false; // Too high

		if (abs(extrapPosWhenScore.x) > APPROX_GOAL_HALF_WIDTH + scoreMargin)
			return false; // Too far to the side

		if (goalTeamOut)
			*goalTeamOut = RS_TEAM_FROM_Y(scoreDirSgn);

		// Ok it's probably gonna score, or at least be very close
		return true;
	} else if (gameMode == GameMode::HOOPS) {

		constexpr float
			APPROX_RIM_HEIGHT = 365;

		float minHeight = APPROX_RIM_HEIGHT + mutators.ballRadius * 1.2f;

		if (ballVel.z < -FLT_EPSILON && ballPos.z < minHeight) {
			if (BallWithinHoopsGoalXYMarginSq(ballPos.x, ballPos.y) < 0) {
				if (goalTeamOut)
					*goalTeamOut = RS_TEAM_FROM_Y(ballPos.y);
				return true; // Already in the net
			}
		}

		float margin = mutators.ballRadius * 1.0f;
		float marginSq = margin * margin;

		float upQuadIntercept;
		float downQuadIntercept;

		// Calculate time to score using quadratic intercept
		{
			float g = mutators.gravity.z;
			if (g > -FLT_EPSILON)
				return false;

			float v = ballVel.z;
			float h = ballPos.z - minHeight;

			float sqrtInput = v * v - 2 * g * h;
			if (sqrtInput > 0) {
				float sqrtOutput = sqrtf(sqrtInput);
				upQuadIntercept = (-v + sqrtOutput) / g;
				downQuadIntercept = (-v - sqrtOutput) / g;
			} else {
				// Never reaches the rim height
				// If started within the hoop, it will stay within the hoop and is therefore scoring
				return BallWithinHoopsGoalXYMarginSq(ballPos.x, ballPos.y) < -marginSq;
			}
		}

		if (upQuadIntercept >= 0) {
			// Ball has to go up before it can fall into the hoop
			// Make sure it cant hit the rim on the way up

			Vec extrapPosUp = ballPos + (ballVel * upQuadIntercept);
			float upMarginSq = BallWithinHoopsGoalXYMarginSq(extrapPosUp.x, extrapPosUp.y);

			float minClearanceMargin = 60 + mutators.ballRadius;

			if (upMarginSq > -marginSq && upMarginSq < (minClearanceMargin * minClearanceMargin))
				return false; // Will probably hit rim
		}

		Vec extrapPosDown = ballPos + (ballVel * downQuadIntercept);
		extrapPosDown.y = abs(extrapPosDown.y);

		{ // Very approximate prediction of backboard bounce
			float wallBounceY = RLConst::ARENA_EXTENT_Y_HOOPS - mutators.ballRadius;
			if (extrapPosDown.y > wallBounceY) {
				float margin = extrapPosDown.y - wallBounceY;
				extrapPosDown.y -= margin * (1 + RLConst::BALL_WORLD_RESTITUTION);
			}
		}

		if (BallWithinHoopsGoalXYMarginSq(extrapPosDown.x, extrapPosDown.y) < -marginSq) {
			if (goalTeamOut)
				*goalTeamOut = RS_TEAM_FROM_Y(extrapPosDown.y);
			return true;
		} else {
			return false;
		}

	} else {
		RS_ERR_CLOSE("Arena::IsBallProbablyGoingIn() is not supported for gamemode " << GAMEMODE_STRS[(int)gameMode]);
		return false;
	}
}

Arena::~Arena() {
	for (Car* car : _cars)
		delete car;
	for (BoostPad* pad : _boostPads)
		delete pad;
	delete ball;
	rs_arena_free(_rs);
}

RS_NS_END
