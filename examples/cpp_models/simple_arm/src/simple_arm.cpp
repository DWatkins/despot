#include "simple_arm.h"

#include <cstdlib>

#include <despot/core/builtin_lower_bounds.h>
#include <despot/core/builtin_policy.h>
#include <despot/core/builtin_upper_bounds.h>
#include <despot/core/particle_belief.h>

using namespace std;

namespace despot {

  /* =============================================================================
   * SimpleState class
   * =============================================================================*/

  SimpleState::SimpleState() {
  }

  SimpleState::~SimpleState() {
  }

  string SimpleState::text() const {
    return "rover position = " + to_string(rover_position) + " switch_status = " +
      to_string(switch_status);
  }

  /* =============================================================================
   * SimpleRockSample class
   * =============================================================================*/

  /* ======
   * Action
   * ======*/

  int SimpleRockSample::NumActions() const {
	//    return 8;
	return 8;
  }

  /* ==============================
   * Deterministic simulative model
   * ==============================*/

  bool SimpleRockSample::Step(State& state, double rand_num, ACT_TYPE action,
                              double& reward, OBS_TYPE& obs) const {
    SimpleState& simple_state = static_cast < SimpleState& >(state);
    int& rover_position = simple_state.rover_position;
    int& effector_position = simple_state.effector_position;
    int& joint1_position = simple_state.joint1_position;
    int& joint2_position = simple_state.joint2_position;
    int& switch_status = simple_state.switch_status;
    int& switch_position = simple_state.switch_position;

	if(action == A_WEST){
	  if(rover_position == LEFT){
		reward = -100;
		obs = O_GOOD;
		return true;
	  } else{
		//reward = -1;
		reward = 0;
		obs = O_GOOD;
		rover_position--;
		effector_position += GRID_INCREMENT;
	  }
	} else if (action == A_EAST){
	  if(rover_position == LEFT){
		//reward = -1;
		reward = 0;
		obs = O_GOOD;
		rover_position++;
		effector_position -= GRID_INCREMENT;
	  } else{
		reward = ((joint1_position + joint2_position) == 0) ? 25 : -100;
		//reward = 0;
		obs = O_GOOD;
		rover_position++;
		effector_position -= GRID_INCREMENT;
	  }
	} else if (action == A_FLIP){
	  int delta = std::abs(switch_position - effector_position);
	  if(switch_status && (delta <= 0)){
		//reward = 50 - 5 * (delta);
		reward = 50;
		obs = O_GOOD;
		switch_status = R_BAD;
	  } else {
		reward = -100;
		obs = O_GOOD;
		return true;
	  }
	} else if (action == A_CHECK){
	  reward = 0;
	  obs = switch_status;
	} else { //moving joints
	  if(action == INC_J1){
		if (joint1_position == (num_joint_positions - 1)){
		  reward = -100;
		  obs = O_GOOD;
		  return true;
		} else {
		  joint1_position++;
		  effector_position += JOINT_INCREMENT;
		  reward = -1 * joint1_position;
		  //reward = 0;
		  obs = O_GOOD;
		}
	  } else if (action == DEC_J1){
		if(joint1_position == 0){
		  reward = -100;
		  obs = O_GOOD;
		  return true;
		} else {
		  reward = -1 * joint1_position;
		  joint1_position--;
		  effector_position -= JOINT_INCREMENT;
		  //reward = 0;
		  obs = O_GOOD;
		}
	  } else if (action == INC_J2){
		if (joint2_position == (num_joint_positions - 1)){
		  reward = -100;
		  obs = O_GOOD;
		  return true;
		} else {
		  joint2_position++;
		  effector_position += JOINT_INCREMENT;
		  reward = -1 * joint2_position;
		  //reward = 0;
		  obs = O_GOOD;
		}

	  } else{ //action is DEC_J2
		if(joint2_position == 0){
		  reward = -100;
		  obs = O_GOOD;
		  return true;
		} else {
		  reward = -1 * joint2_position;
		  joint2_position--;
		  effector_position -= JOINT_INCREMENT;
		  //reward =0;
		  obs = O_GOOD;
		}
	  }
	}
	if(rover_position == EXIT) return true;
	return false;
  }

  /* ================================================
   * Functions related to beliefs and starting states
   * ================================================*/

  double SimpleRockSample::ObsProb(OBS_TYPE obs, const State& state,
                                   ACT_TYPE action) const {
	if(action == A_CHECK){
	  const SimpleState& simple_state = static_cast<const SimpleState&>(state);
	  int switch_status = simple_state.switch_status;
	  return (obs == switch_status) ? 1 : 0;
	}
	return 1;
  }

  State* SimpleRockSample::CreateStartState(string type) const {
	//return new SimpleState(1, 0, 0, R_GOOD, DEBUG_SWITCH_INIT_POSITION);
	return new SimpleState(1, 0, 0, R_GOOD, Random::RANDOM.NextInt(TOTAL_SWITCH_POSITIONS));
  }

  Belief* SimpleRockSample::InitialBelief(const State* start, string type) const {
    if (type == "DEFAULT" || type == "PARTICLE") {
      vector<State*> particles;

	  const SimpleState& simple_state = static_cast<const SimpleState&>(*start);

	  SimpleState* good_rock = static_cast<SimpleState*>(Allocate(-1, 0.5));
      good_rock->rover_position = 1;
      good_rock->effector_position = 0;
      good_rock->joint1_position = 0;
      good_rock->joint2_position = 0;
      good_rock->switch_position = simple_state.switch_position;
      good_rock->switch_status = O_GOOD;
      particles.push_back(good_rock);

	  SimpleState* bad_rock = static_cast<SimpleState*>(Allocate(-1, 0.5));
      bad_rock->rover_position = 1;
      good_rock->effector_position = 0;
      good_rock->joint1_position = 0;
      good_rock->joint2_position = 0;
      good_rock->switch_position = simple_state.switch_position;
      bad_rock->switch_status = O_BAD;
      particles.push_back(bad_rock);

      return new ParticleBelief(particles, this);
    } else {
      cerr << "[SimpleRockSample::InitialBelief] Unsupported belief type: " << type << endl;
      exit(1);
    }
  }

  /* ========================
   * Bound-related functions.
   * ========================*/
  /*
    Note: in the following bound-related functions, only GetMaxReward() and
    GetBestAction() functions are required to be implemented. The other
    functions (or classes) are for custom bounds. You don't need to write them
    if you don't want to use your own custom bounds. However, it is highly
    recommended that you build the bounds based on the domain knowledge because
    it often improves the performance. Read the tutorial for more details on how
    to implement custom bounds.
  */
  double SimpleRockSample::GetMaxReward() const {
    return 50;
  }


  ScenarioUpperBound* SimpleRockSample::CreateScenarioUpperBound(string name,
                                                                 string particle_bound_name) const {
    ScenarioUpperBound* bound = NULL;
    if (name == "TRIVIAL" || name == "DEFAULT") {
      bound = new TrivialParticleUpperBound(this);
    } else {
      cerr << "Unsupported base upper bound: " << name << endl;
      exit(0);
    }
    return bound;
  }

  ValuedAction SimpleRockSample::GetBestAction() const {
    return ValuedAction(A_CHECK, 0);
  }

  class SimpleRockSampleEastPolicy: public DefaultPolicy {
  public:
    enum { // action
	  A_FLIP = 0, A_EAST = 1, A_WEST = 2, A_CHECK = 3, INC_J1 = 4, INC_J2 = 5, DEC_J1 = 6, DEC_J2 = 7
    };
    SimpleRockSampleEastPolicy(const DSPOMDP* model, ParticleLowerBound* bound) :
      DefaultPolicy(model, bound) {
    }

    ACT_TYPE Action(const vector<State*>& particles, RandomStreams& streams,
                    History& history) const {
      return A_EAST; // move east
    }
  };

  ScenarioLowerBound* SimpleRockSample::CreateScenarioLowerBound(string name,
                                                                 string particle_bound_name) const {
    ScenarioLowerBound* bound = NULL;
    if (name == "TRIVIAL" || name == "DEFAULT") {
      bound = new TrivialParticleLowerBound(this);
    } else if (name == "EAST") {
      bound = new SimpleRockSampleEastPolicy(this,
                                             CreateParticleLowerBound(particle_bound_name));
	}
	else {
      cerr << "Unsupported lower bound algorithm: " << name << endl;
      exit(0);
    }
    return bound;
  }

  /* =================
   * Memory management
   * =================*/

  State* SimpleRockSample::Allocate(int state_id, double weight) const {
    SimpleState* state = memory_pool_.Allocate();
    state->state_id = state_id;
    state->weight = weight;
    return state;
  }

  State* SimpleRockSample::Copy(const State* particle) const {
    SimpleState* state = memory_pool_.Allocate();
    *state = *static_cast<const SimpleState*>(particle);
    state->SetAllocated();
    return state;
  }

  void SimpleRockSample::Free(State* particle) const {
    memory_pool_.Free(static_cast<SimpleState*>(particle));
  }

  int SimpleRockSample::NumActiveParticles() const {
    return memory_pool_.num_allocated();
  }

  /* =======
   * Display
   * =======*/

  void SimpleRockSample::PrintState(const State& state, ostream& out) const {
    const SimpleState& simple_state = static_cast<const SimpleState&>(state);

    out << "Rover = " << simple_state.rover_position << ";\n"
		<< "Grid Increment = " << GRID_INCREMENT << "\n"
		<< "Joint 1 = " << simple_state.joint1_position << "\n"
        << "Joint 2 = " << simple_state.joint2_position << "\n"
        << "Joint Increment = " << JOINT_INCREMENT << "\n"
		<< "Max Joint Positions = " << num_joint_positions << "\n"
		<< "End Effector = " << simple_state.effector_position << "\n"
        << "Switch Status = " << (simple_state.switch_status ? "GOOD" : "BAD") << "\n"
        << "Switch_Position = " << simple_state.switch_position << "\n"
        << "Delta = " << simple_state.switch_position - simple_state.effector_position << "\n";

  }

  void SimpleRockSample::PrintObs(const State& state, OBS_TYPE observation,
                                  ostream& out) const {
    out << (observation ? "GOOD" : "BAD") << endl;
  }

  void SimpleRockSample::PrintBelief(const Belief& belief, ostream& out) const {
    const vector<State*>& particles =
      static_cast<const ParticleBelief&>(belief).particles();

    double switch_status = 0;
    vector<double> pos_probs(3);
    for (int i = 0; i < particles.size(); i++) {
      State* particle = particles[i];
      const SimpleState* state = static_cast<const SimpleState*>(particle);
      switch_status += state->switch_status * particle->weight;
      pos_probs[state->rover_position] += particle->weight;
    }

    out << "Rock belief: " << switch_status << endl;

    out << "Position belief: "
        << "LEFT" << ":" << pos_probs[0] << " "
        << "MIDDLE" << ":" << pos_probs[1] << " "
        << "EXIT" << ":" << pos_probs[2] << endl;
  }

  void SimpleRockSample::PrintAction(ACT_TYPE action, ostream& out) const {
    if (action == A_FLIP)
      out << "Flip" << endl;
    if (action == A_CHECK)
      out << "Check" << endl;
    if (action == A_EAST)
      out << "East " << endl;
    if (action == A_WEST)
      out << "West" << endl;
    if (action == INC_J1)
      out << "Inc j1" << endl;
    if (action == DEC_J1)
      out << "Dec j1" << endl;
    if (action == INC_J2)
      out << "Inc j2" << endl;
    if (action == DEC_J2)
      out << "Dec j2" << endl;
  }

} // namespace despot
