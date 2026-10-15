#include "Math.h"

Vec RLGC::Math::RandVec(Vec min, Vec max) {
	return Vec(
		RocketSim::Math::RandFloat(min.x, max.x),
		RocketSim::Math::RandFloat(min.y, max.y),
		RocketSim::Math::RandFloat(min.z, max.z)
	);
}
