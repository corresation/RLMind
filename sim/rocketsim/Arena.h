#pragma once
#include "Math.h"

// RocketSim 2's Arena/Car/Ball/BoostPad, with RocketSim v3 (Rust) doing the physics underneath
// States are pulled from v3 once per Step() and cached here, so GetState() is cheap

RS_NS_START

enum class GameMode : byte {
	SOCCAR,
	HOOPS,
	HEATSEEKER,
	SNOWDAY,
	DROPSHOT,
	THE_VOID,
};

constexpr const char* GAMEMODE_STRS[] = {
	"soccar",
	"hoops",
	"heatseeker",
	"snowday",
	"dropshot",
	"void"
};

enum class Team : byte {
	BLUE = 0,
	ORANGE = 1
};

#define RS_OPPOSITE_TEAM(team) ((team) == Team::BLUE ? Team::ORANGE : Team::BLUE)
#define RS_TEAM_FROM_Y(y) ((y) < 0 ? Team::BLUE : Team::ORANGE)

namespace RLConst {
	constexpr float
		CAR_MAX_SPEED = 2300,
		BALL_MAX_SPEED = 6000,
		CAR_MAX_ANG_SPEED = 5.5f,
		BALL_MAX_ANG_SPEED = 6,
		BOOST_MAX = 100,
		SUPERSONIC_START_SPEED = 2200,
		GRAVITY_Z = -650,
		ARENA_EXTENT_X = 4096,
		ARENA_EXTENT_Y = 5120,
		ARENA_HEIGHT = 2048,
		ARENA_EXTENT_Y_HOOPS = 3581,
		BALL_REST_Z = 93.15f,
		CAR_SPAWN_REST_Z = 17,
		BALL_WORLD_RESTITUTION = 0.6f;
}

struct PhysState {
	// Position in world space (UU)
	Vec pos = {};

	// Rotation matrix (column-major)
	RotMat rotMat = RotMat::GetIdentity();

	// Linear velocity
	Vec vel = {};

	// Angular velocity (rad/s)
	Vec angVel = {};
};

// Stores all control inputs to a car
struct CarControls {
	// Driving control
	float throttle, steer;

	// Air orientation control
	float pitch, yaw, roll;

	// Boolean action inputs
	bool jump, boost, handbrake;

	CarControls() {
		memset(this, 0, sizeof(CarControls));
	}

	// Makes all values range-valid (clamps from -1 to 1)
	void ClampFix() {
		throttle = RS_CLAMP(throttle, -1, 1);
		steer = RS_CLAMP(steer, -1, 1);
		pitch = RS_CLAMP(pitch, -1, 1);
		yaw = RS_CLAMP(yaw, -1, 1);
		roll = RS_CLAMP(roll, -1, 1);
	}
};

struct BallHitInfo {
	// If false, all other fields within this struct should not be trusted
	bool isValid = false;

	Vec relativePosOnBall; // Position of the hit relative to the ball's position
	Vec ballPos; // World position of the ball when the hit occured
	Vec extraHitVel; // Extra velocity added to base collision velocity

	// Arena tick count when the hit occured
	uint64_t tickCountWhenHit = ~0ULL;
};

struct CarState : public PhysState {
	// Incremented every update, reset when SetState() is called
	uint64_t updateCounter = 0;

	// True if 3 or more wheels have contact
	bool isOnGround = true;

	// Whether each of the 4 wheels have contact, first two are front
	bool wheelsWithContact[4] = {};

	// Whether we jumped to get into the air
	// Can be false while airborne, if we left the ground with a flip reset
	bool hasJumped = false;

	// True if we have double jumped and are still in the air
	bool hasDoubleJumped = false;

	// True if we are in the air, and (have flipped or are currently flipping)
	bool hasFlipped = false;

	// Relative torque direction of the flip, forward flip will have positive Y
	Vec flipRelTorque = { 0, 0, 0 };

	float jumpTime = 0; // Time since the current jump phase began
	float flipTime = 0; // Time since the current flip started, keeps counting until landing

	// True during a flip (not an auto-flip, and not after a flip)
	bool isFlipping = false;

	// True during a jump
	bool isJumping = false;

	// Total time spent in the air
	float airTime = 0;

	// Time spent in the air once !isJumping, if we never jumped it is 0
	float airTimeSinceJump = 0;

	// Goes from 0 to 100
	float boost = 100 / 3.f;

	// Time spent continuously boosting
	float timeSpentBoosting = 0;

	bool isSupersonic = false;

	// Grace timer for staying supersonic below the start speed
	float supersonicTime = 0;

	// Smoothed handbrake, 0 to 1
	float handbrakeVal = 0;

	bool isAutoFlipping = false;
	float autoFlipTimer = 0; // Counts down when auto-flipping
	float autoFlipTorqueScale = 0;

	struct {
		bool hasContact = false;
		Vec contactNormal;
	} worldContact;

	struct {
		uint32_t otherCarID = 0;
		float cooldownTimer = 0;
	} carContact;

	bool isDemoed = false;
	float demoRespawnTimer = 0;

	BallHitInfo ballHitInfo = BallHitInfo();

	// Controls from the last tick
	CarControls lastControls = CarControls();

	CarState() : PhysState() {
		pos.z = RLConst::CAR_SPAWN_REST_Z;
	}

	// Returns true if the car is currently able to jump, double-jump, or flip
	bool HasFlipOrJump() const {
		return isOnGround || (!hasFlipped && !hasDoubleJumped && airTimeSinceJump < 1.25f);
	}

	// Returns true if we currently have a flip, and got it from a flip reset (instead of a jump)
	bool HasFlipReset() const {
		return !isOnGround && HasFlipOrJump() && !hasJumped;
	}

	// Returns true if we are in the air, and became airborne from a flip reset
	bool GotFlipReset() const {
		return !isOnGround && !hasJumped;
	}
};

struct BallState : public PhysState {
	// Incremented every update, reset when SetState() is called
	// Used for telling if a stateset occured
	uint64_t updateCounter = 0;

	BallState() : PhysState() {
		pos.z = RLConst::BALL_REST_Z;
	}
};

struct BoostPadConfig {
	Vec pos;
	bool isBig = false;
};

struct BoostPadState {
	bool isActive = true;
	float cooldown = 0;
};

class Arena;

class Car {
public:
	Team team;

	// Each car is given a unique ID when created by the arena, always >0
	uint32_t id;

	// Controls to simulate the car with
	CarControls controls;

	CarState _internalState;
	int _idx;
	Arena* _arena;

	CarState GetState() const { return _internalState; }
	void SetState(const CarState& state);
};

class Ball {
public:
	BallState _internalState;
	Arena* _arena;

	BallState GetState() const { return _internalState; }
	void SetState(const BallState& state);
};

class BoostPad {
public:
	BoostPadConfig config;
	BoostPadState _internalState;
	int _idx;
	Arena* _arena;

	BoostPadState GetState() const { return _internalState; }
	void SetState(const BoostPadState& state);
};

typedef std::function<void(class Arena* arena, Car* bumper, Car* victim, bool isDemo, void* userInfo)> CarBumpEventFn;

struct MutatorConfig {
	Vec gravity;
	float ballRadius;
	float goalBaseThresholdY;
};

class Arena {
public:
	GameMode gameMode;
	std::vector<Car*> _cars; // In the order they were added
	Ball* ball;
	std::vector<BoostPad*> _boostPads;

	uint64_t tickCount = 0;
	float tickTime = 1 / 120.f;

	MutatorConfig _mutatorConfig;

	struct {
		CarBumpEventFn func = NULL;
		void* userInfo = NULL;
	} _carBumpCallback;

	void* _rs; // The RocketSim v3 arena

	RSAPI static Arena* Create(GameMode gameMode);

	RSAPI Car* AddCar(Team team);

	float GetTickRate() const {
		return 1 / tickTime;
	}

	const std::vector<Car*>& GetCars() { return _cars; }
	const std::vector<BoostPad*>& GetBoostPads() { return _boostPads; }

	RSAPI void SetCarBumpCallback(CarBumpEventFn callbackFn, void* userInfo = NULL);

	RSAPI void Step(int ticksToSimulate = 1);

	RSAPI void ResetToRandomKickoff(int seed = -1);

	// Returns true if the ball is in the net
	RSAPI bool IsBallScored() const;

	// Returns true if the ball is probably going in, does not account for wall or ceiling bounces
	RSAPI bool IsBallProbablyGoingIn(float maxTime = 2.f, float extraMargin = 0, Team* goalTeamOut = NULL) const;

	RSAPI ~Arena();

	void _PullState();

private:
	Arena() = default;
};

RS_NS_END
