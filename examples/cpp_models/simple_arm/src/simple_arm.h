#ifndef SIMPLEHALLWAYROCKSAMPLE_H
#define SIMPLEHALLWAYROCKSAMPLE_H

#include <despot/interface/pomdp.h>
#include <despot/core/mdp.h>

#define JOINT_POSITIONS 3
#define JOINT_INCREMENT 2
#define GRID_INCREMENT 3
#define TOTAL_SWITCH_POSITIONS 12

namespace despot {

/* =============================================================================
 * SimpleState class
 * =============================================================================*/

class SimpleState: public State {
public:
	int rover_position; // positions are numbered 0, 1, 2 from left to right
  int effector_position; //location of end effector, calculated from joint values
  int joint1_position; // value of the first joint of the arm,
  int joint2_position; // value of the second joint of the arm,
	int switch_status; // 1 is good, and 0 is bad
  int switch_position = 0; //position of the switch

	SimpleState();
	SimpleState(int _rover_position, int  _joint1_position, int _joint2_position, int _switch_status, int _switch_position) :
        rover_position(_rover_position),
        joint1_position(_joint1_position),
        joint2_position(_joint2_position),
        switch_status(_switch_status),
        switch_position(_switch_position)
    {
      effector_position = JOINT_INCREMENT * (joint1_position + joint2_position) - (1-rover_position) * GRID_INCREMENT;
    }
	~SimpleState();

	std::string text() const;
};

/* =============================================================================
 * SimpleRockSample class
 * =============================================================================*/

class SimpleRockSample: public DSPOMDP {

protected:

  const  int num_joint_positions = JOINT_POSITIONS;
  const int joint_increments = JOINT_INCREMENT;
  int switch_init_pos;
protected:
	mutable MemoryPool<SimpleState> memory_pool_;

	std::vector<SimpleState*> states_;

	mutable std::vector<ValuedAction> mdp_policy_;

public:
	enum { // actions, A_CHECK should always be last
        A_FLIP = 0, A_EAST = 1, A_WEST = 2, A_CHECK = 3, INC_J1 = 4, INC_J2 = 5, DEC_J1 = 6, DEC_J2 = 7
	};
	enum { // observation
        O_BAD = 0, O_GOOD = 1, O_NONE = 3
	};
	enum { // button status
		R_BAD = 0, R_GOOD = 1
	};
	enum { // rover position
        LEFT = 0, MIDDLE = 1, EXIT = 2
	};

public:
	SimpleRockSample(){
    switch_init_pos = Random::RANDOM.NextInt(TOTAL_SWITCH_POSITIONS);
  }

	/* Returns total number of actions.*/
	int NumActions() const;

	/* Deterministic simulative model.*/
	bool Step(State& state, double rand_num, ACT_TYPE action, double& reward,
		OBS_TYPE& obs) const;

	/* Functions related to beliefs and starting states.*/
	double ObsProb(OBS_TYPE obs, const State& state, ACT_TYPE action) const;
	State* CreateStartState(std::string type = "DEFAULT") const;
	Belief* InitialBelief(const State* start, std::string type = "DEFAULT") const;

	/* Bound-related functions.*/
	double GetMaxReward() const;
	ScenarioUpperBound* CreateScenarioUpperBound(std::string name = "DEFAULT",
		std::string particle_bound_name = "DEFAULT") const;
	ValuedAction GetBestAction() const;
	ScenarioLowerBound* CreateScenarioLowerBound(std::string name = "DEFAULT",
		std::string particle_bound_name = "DEFAULT") const;

	/* Memory management.*/
	State* Allocate(int state_id, double weight) const;
	State* Copy(const State* particle) const;
	void Free(State* particle) const;
	int NumActiveParticles() const;

	/* Display.*/
	void PrintState(const State& state, std::ostream& out = std::cout) const;
	void PrintBelief(const Belief& belief, std::ostream& out = std::cout) const;
	void PrintObs(const State& state, OBS_TYPE observation,
		std::ostream& out = std::cout) const;
	void PrintAction(ACT_TYPE action, std::ostream& out = std::cout) const;
};

} // namespace despot

#endif
