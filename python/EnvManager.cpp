#include "EnvManager.h"
#include "DARTHelper.h"
#include "BVH.h"
#include "Verbose.h"
#include <omp.h>

EnvManager::
EnvManager(std::string meta_file,int num_envs)
	:mNumEnvs(num_envs)
{
	dart::math::seedRand();
	
	// OPTIMIZATION 1: Set OpenMP thread count based on available cores
	// Using all available threads can cause contention; often better to match physical cores
	int max_threads = omp_get_max_threads();
	int optimal_threads = std::min(mNumEnvs, max_threads);
	omp_set_num_threads(optimal_threads);
	
	// OPTIMIZATION 2: Set OpenMP scheduling for better load balancing
	// Dynamic scheduling helps when environments have varying workloads
	omp_set_schedule(omp_sched_dynamic, 1);
	
	for(int i = 0;i<mNumEnvs;i++){
		mEnvs.push_back(new MASS::Environment());
		MASS::Environment* env = mEnvs.back();
		env->Initialize(meta_file,false);
	}
	
	muscle_torque_cols = mEnvs[0]->GetMuscleTorques().rows();
	tau_des_cols = mEnvs[0]->GetDesiredTorques().rows();
	
	// OPTIMIZATION 3: Pre-allocate all matrices once
	// This avoids repeated memory allocations during simulation
	mEoe.resize(mNumEnvs);
	mRewards.resize(mNumEnvs);
	mStates.resize(mNumEnvs, GetNumState());
	mMuscleTorques.resize(mNumEnvs, muscle_torque_cols);
	mDesiredTorques.resize(mNumEnvs, tau_des_cols);
	
	// Zero-initialize to avoid undefined behavior
	mEoe.setZero();
	mRewards.setZero();
	mStates.setZero();
	mMuscleTorques.setZero();
	mDesiredTorques.setZero();
}

int
EnvManager::
GetNumState()
{
	return mEnvs[0]->GetNumState();
}
int
EnvManager::
GetNumAction()
{
	return mEnvs[0]->GetNumAction();
}
int
EnvManager::
GetSimulationHz()
{
	return mEnvs[0]->GetSimulationHz();
}
int
EnvManager::
GetControlHz()
{
	return mEnvs[0]->GetControlHz();
}
int
EnvManager::
GetNumSteps()
{
	return mEnvs[0]->GetNumSteps();
}
bool
EnvManager::
UseMuscle()
{
	return mEnvs[0]->GetUseMuscle();
}
void
EnvManager::
Step(int id)
{
	mEnvs[id]->Step();
}
void
EnvManager::
Reset(bool RSI,int id)
{
	mEnvs[id]->Reset(RSI);
}
bool
EnvManager::
IsEndOfEpisode(int id)
{
	return mEnvs[id]->IsEndOfEpisode();
}

double 
EnvManager::
GetReward(int id)
{
	return mEnvs[id]->GetReward();
}

void
EnvManager::
Steps(int num)
{
	// OPTIMIZATION 4: Use schedule(dynamic) for better load balancing
	// when different environments may terminate at different times
#pragma omp parallel for schedule(dynamic)
	for (int id = 0; id < mNumEnvs; ++id)
	{
		for(int j = 0; j < num; j++)
			mEnvs[id]->Step();
	}
}

void
EnvManager::
StepsAtOnce()
{
	int num = this->GetNumSteps();
	
	// OPTIMIZATION 5: Use static scheduling when workload is uniform
	// Static has less overhead than dynamic for uniform workloads
#pragma omp parallel for schedule(static)
	for (int id = 0; id < mNumEnvs; ++id)
	{
		for(int j = 0; j < num; j++)
			mEnvs[id]->Step();
	}
}

void
EnvManager::
Resets(bool RSI)
{
	// OPTIMIZATION 6: Parallelize resets as well
	// Reset can involve significant computation
#pragma omp parallel for schedule(static)
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mEnvs[id]->Reset(RSI);
	}
}

const Eigen::VectorXd&
EnvManager::
IsEndOfEpisodes()
{
	// OPTIMIZATION 7: Parallelize episode end checks
	// These are lightweight but parallelization still helps with many envs
#pragma omp parallel for schedule(static)
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mEoe[id] = (double)mEnvs[id]->IsEndOfEpisode();
	}

	return mEoe;
}

const Eigen::MatrixXd&
EnvManager::
GetStates()
{
	// OPTIMIZATION 8: Parallelize state gathering
#pragma omp parallel for schedule(static)
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mStates.row(id) = mEnvs[id]->GetState().transpose();
	}

	return mStates;
}

void
EnvManager::
SetActions(const Eigen::MatrixXd& actions)
{
	// OPTIMIZATION 9: Parallelize action setting
#pragma omp parallel for schedule(static)
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mEnvs[id]->SetAction(actions.row(id).transpose());
	}
}

const Eigen::VectorXd&
EnvManager::
GetRewards()
{
	// OPTIMIZATION 10: Parallelize reward computation
#pragma omp parallel for schedule(static)
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mRewards[id] = mEnvs[id]->GetReward();
	}
	return mRewards;
}

const Eigen::MatrixXd&
EnvManager::
GetMuscleTorques()
{
#pragma omp parallel for schedule(static)
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mMuscleTorques.row(id) = mEnvs[id]->GetMuscleTorques();
	}
	return mMuscleTorques;
}

const Eigen::MatrixXd&
EnvManager::
GetDesiredTorques()
{
#pragma omp parallel for schedule(static)
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mDesiredTorques.row(id) = mEnvs[id]->GetDesiredTorques();
	}
	return mDesiredTorques;
}

void
EnvManager::
SetActivationLevels(const Eigen::MatrixXd& activations)
{
	// OPTIMIZATION 11: Parallelize activation level setting
#pragma omp parallel for schedule(static)
	for (int id = 0; id < mNumEnvs; ++id)
		mEnvs[id]->SetActivationLevels(activations.row(id));
}

void
EnvManager::
ComputeMuscleTuples()
{
	// First pass: count total tuples (sequential, quick)
	int n = 0;
	int rows_JtA = 0;
	int rows_tau_des = 0;
	int rows_L = 0;
	int rows_b = 0;

	for(int id = 0; id < mNumEnvs; id++)
	{
		auto& tps = mEnvs[id]->GetMuscleTuples();
		n += tps.size();
		if(tps.size() != 0)
		{
			rows_JtA = tps[0].JtA.rows();
			rows_tau_des = tps[0].tau_des.rows();
			rows_L = tps[0].L.rows();
			rows_b = tps[0].b.rows();
		}
	}
	
	// OPTIMIZATION 12: Pre-allocate output matrices
	mMuscleTuplesJtA.resize(n, rows_JtA);
	mMuscleTuplesTauDes.resize(n, rows_tau_des);
	mMuscleTuplesL.resize(n, rows_L);
	mMuscleTuplesb.resize(n, rows_b);

	// Compute offsets for each environment (for parallel filling)
	std::vector<int> offsets(mNumEnvs + 1);
	offsets[0] = 0;
	for(int id = 0; id < mNumEnvs; id++)
	{
		offsets[id + 1] = offsets[id] + mEnvs[id]->GetMuscleTuples().size();
	}

	// OPTIMIZATION 13: Parallel tuple copying with pre-computed offsets
#pragma omp parallel for schedule(static)
	for(int id = 0; id < mNumEnvs; id++)
	{
		auto& tps = mEnvs[id]->GetMuscleTuples();
		int offset = offsets[id];
		for(size_t j = 0; j < tps.size(); j++)
		{
			mMuscleTuplesJtA.row(offset + j) = tps[j].JtA;
			mMuscleTuplesTauDes.row(offset + j) = tps[j].tau_des;
			mMuscleTuplesL.row(offset + j) = tps[j].L;
			mMuscleTuplesb.row(offset + j) = tps[j].b;
		}
		tps.clear();
	}
}

const Eigen::MatrixXd&
EnvManager::
GetMuscleTuplesJtA()
{
	return mMuscleTuplesJtA;
}
const Eigen::MatrixXd&
EnvManager::
GetMuscleTuplesTauDes()
{
	return mMuscleTuplesTauDes;
}
const Eigen::MatrixXd&
EnvManager::
GetMuscleTuplesL()
{
	return mMuscleTuplesL;
}
const Eigen::MatrixXd&
EnvManager::
GetMuscleTuplesb()
{
	return mMuscleTuplesb;
}

// =============================================================================
// BVH Data Extraction for Pre-training
// =============================================================================

int
EnvManager::
GetBVHFrameCount()
{
	return mEnvs[0]->GetCharacter()->GetBVH()->GetNumFrames();
}

double
EnvManager::
GetBVHFrameTime()
{
	return mEnvs[0]->GetCharacter()->GetBVH()->GetTimeStep();
}

double
EnvManager::
GetBVHMaxTime()
{
	return mEnvs[0]->GetCharacter()->GetBVH()->GetMaxTime();
}

Eigen::VectorXd
EnvManager::
GetTargetPositions(double t)
{
	double dt = 1.0 / GetControlHz();
	return mEnvs[0]->GetCharacter()->GetTargetPositions(t, dt);
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
EnvManager::
GetTargetPosAndVel(double t)
{
	double dt = 1.0 / GetControlHz();
	return mEnvs[0]->GetCharacter()->GetTargetPosAndVel(t, dt);
}

bool
EnvManager::
ReloadBVH(const std::string& path, bool cyclic)
{
	return mEnvs[0]->GetCharacter()->ReloadBVH(path, cyclic);
}

// =============================================================================
// Verbose Control
// =============================================================================

void SetVerbose(bool verbose)
{
	MASS::gVerbose = verbose;
}

bool GetVerbose()
{
	return MASS::gVerbose;
}

PYBIND11_MODULE(pymss, m)
{
	// Module-level verbose control functions
	m.def("set_verbose", &SetVerbose, "Set C++ verbose output (true/false)");
	m.def("get_verbose", &GetVerbose, "Get C++ verbose output setting");
	
	py::class_<EnvManager>(m, "pymss")
		.def(py::init<std::string,int>())
		.def("GetNumState",&EnvManager::GetNumState)
		.def("GetNumAction",&EnvManager::GetNumAction)
		.def("GetSimulationHz",&EnvManager::GetSimulationHz)
		.def("GetControlHz",&EnvManager::GetControlHz)
		.def("GetNumSteps",&EnvManager::GetNumSteps)
		.def("UseMuscle",&EnvManager::UseMuscle)
		.def("Step",&EnvManager::Step)
		.def("Reset",&EnvManager::Reset)
		.def("IsEndOfEpisode",&EnvManager::IsEndOfEpisode)
		.def("GetReward",&EnvManager::GetReward)
		.def("Steps",&EnvManager::Steps)
		.def("StepsAtOnce",&EnvManager::StepsAtOnce)
		.def("Resets",&EnvManager::Resets)
		.def("IsEndOfEpisodes",&EnvManager::IsEndOfEpisodes)
		.def("GetStates",&EnvManager::GetStates)
		.def("SetActions",&EnvManager::SetActions)
		.def("GetRewards",&EnvManager::GetRewards)
		.def("GetNumTotalMuscleRelatedDofs",&EnvManager::GetNumTotalMuscleRelatedDofs)
		.def("GetNumMuscles",&EnvManager::GetNumMuscles)
		.def("GetMuscleTorques",&EnvManager::GetMuscleTorques)
		.def("GetDesiredTorques",&EnvManager::GetDesiredTorques)
		.def("SetActivationLevels",&EnvManager::SetActivationLevels)
		.def("ComputeMuscleTuples",&EnvManager::ComputeMuscleTuples)
		.def("GetMuscleTuplesJtA",&EnvManager::GetMuscleTuplesJtA)
		.def("GetMuscleTuplesTauDes",&EnvManager::GetMuscleTuplesTauDes)
		.def("GetMuscleTuplesL",&EnvManager::GetMuscleTuplesL)
		.def("GetMuscleTuplesb",&EnvManager::GetMuscleTuplesb)
		// BVH extraction methods for pre-training
		.def("GetBVHFrameCount",&EnvManager::GetBVHFrameCount)
		.def("GetBVHFrameTime",&EnvManager::GetBVHFrameTime)
		.def("GetBVHMaxTime",&EnvManager::GetBVHMaxTime)
		.def("GetTargetPositions",&EnvManager::GetTargetPositions)
		.def("GetTargetPosAndVel",&EnvManager::GetTargetPosAndVel)
		.def("ReloadBVH",&EnvManager::ReloadBVH, py::arg("path"), py::arg("cyclic") = false);
}