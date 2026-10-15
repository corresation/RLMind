#pragma once
#include "Framework.h"

namespace RLGC {
	namespace Math {
		Vec RandVec(Vec min, Vec max);

		constexpr float VelToKPH(float vel) {
			return vel / (250.f / 9.f);
		}

		constexpr float KPHToVel(float vel) {
			return vel * (250.f / 9.f);
		}
	}
}