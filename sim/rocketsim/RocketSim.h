#pragma once
#include "Arena.h"
#include "GameEventTracker.h"

RS_NS_START

enum class RocketSimStage : byte {
	UNINITIALIZED,
	INITIALIZING,
	INITIALIZED
};

void Init(std::filesystem::path collisionMeshesFolder, bool silent = false);

RocketSimStage GetStage();

RS_NS_END
