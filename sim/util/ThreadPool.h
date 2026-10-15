#pragma once
#include "Framework.h"

#include "thread_pool.h"

namespace RLGC {
	// Modified version of https://stackoverflow.com/questions/26516683/reusing-thread-in-loop-c
	struct ThreadPool {

		dp::thread_pool<>* _tp;

		ThreadPool() {
			_tp = new dp::thread_pool();
		}

		RG_NO_COPY(ThreadPool);

		~ThreadPool() {
			delete _tp;
		}

		template <typename Function, typename... Args> requires std::invocable<Function, Args...>
		void StartJobAsync(Function&& func, Args &&...args) {
			_tp->enqueue_detach(func, args...);
		}

		void StartBatchedJobs(std::function<void(int)> func, int num, bool async) {

			// One job per thread running a range of indices, queueing a job per index was slow
			int numJobs = RS_MIN(num, GetNumThreads());
			for (int job = 0; job < numJobs; job++) {
				int start = num * job / numJobs, end = num * (job + 1) / numJobs;
				StartJobAsync([func, start, end] {
					for (int i = start; i < end; i++)
						func(i);
				});
			}

			if (!async)
				WaitUntilDone();
		}

		void WaitUntilDone() {
			_tp->wait_for_tasks();
		}

		int GetNumThreads() const {
			return _tp->size();
		}
	};

	extern ThreadPool g_ThreadPool;
}