#include "simple_arm.h"

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

    if(rover_position != EXIT){
      if(action == A_WEST){
        if(rover_position == LEFT){
          reward = -100;
          obs = O_NONE;
          return true;
        } else{
          reward = 0;
          obs = O_NONE;
          rover_position--;
          effector_position += GRID_INCREMENT;
        }
      } else if (action == A_EAST){
        if(rover_position == LEFT){
          reward = 0;
          obs = O_NONE;
          rover_position++;
          effector_position--;
        } else{
          reward = 0;
          obs = O_NONE;
          rover_position++;
          effector_position -= GRID_INCREMENT;
        }
      } else if (action == A_FLIP){
        int delta = switch_position - effector_position;
        if(switch_status && (delta < 2)){
          reward = 25 - 1 * (delta);
          obs = O_NONE;
          switch_status = R_BAD;
        } else {
          reward = -100;
          obs = O_NONE;
        }
      } else if (action == A_CHECK){
        reward = 0;
        obs = switch_status;
      } else { //moving joints
        if(action == INC_J1){
          if (joint1_position = JOINT_POSITIONS - 1){
            reward = -100;
            obs = O_NONE;
          } else {
            joint1_position++;
            effector_position += JOINT_INCREMENT;
            reward = -2 * joint1_position;
            obs = O_NONE;
              }
        } else if (action == DEC_J1){
          if(joint1_position == 0){
            reward = -100;
            obs = O_NONE;
          } else {
            joint1_position--;
            effector_position -= JOINT_INCREMENT;
            reward = -2 * joint1_position;
            obs = O_NONE;
              }
        } else if (action == INC_J2){
          if (joint2_position = JOINT_POSITIONS - 1){
            reward = -100;
            obs = O_NONE;
          } else {
            joint2_position++;
            effector_position += JOINT_INCREMENT;
            reward = -2 * joint2_position;
            obs = O_NONE;
              }

        } else{ //action is DEC_J2
          if(joint2_position == 0){
            reward = -100;
            obs = O_NONE;
          } else {
            joint2_position--;
            effector_position -= JOINT_INCREMENT;
            reward = -2 * joint2_position;
            obs = O_NONE;
              }
        }
      }
    } else {
      reward = 10;
      obs = O_NONE;
      return true;
    }
    /*
      if (rover_position == LEFT) {
      if (action == A_FLIP) {
      reward = (switch_status == R_GOOD) ? 10 : -10;
      obs = O_GOOD;
      switch_status = R_BAD;
      } else if (action == A_CHECK) {
      reward = 0;
      // when the rover at LEFT, its observation is correct with probability 1
      obs = (switch_status == R_GOOD) ? O_GOOD : O_BAD;  
      } else if (action == A_WEST) {
      reward = -100;
      // moving does not incur observation, setting a default observation 
      // note that we can also set the default observation to O_BAD, as long
      // as it is consistent.
      obs = O_GOOD;
      return true; // Moving off the grid terminates the task. 
      } else { // moving EAST
      reward = 0;
      // moving does not incur observation, setting a default observation
      obs = O_GOOD;
      rover_position++;
      }
      } else {
      if (action == A_FLIP) {
      reward = -100;
      // moving does not incur observation, setting a default observation 
      obs = O_GOOD;
      return true; // sampling in the grid where there is no rock terminates the task
      } else if (action == A_CHECK) {
      reward = 0;
      // when the rover is at MIDDLE, its observation is correct with probability 0.8
      obs =  (rand_num > 0.20) ? switch_status : (1 - switch_status);
      } else if (action == A_WEST) {
      reward = -1;
      // moving does not incur observation, setting a default observation 
      obs = O_GOOD;
      rover_position--;
      } else { //moving EAST to exit
      reward = 0;
      obs = O_GOOD;
      rover_position++;
      }
      }
      if(rover_position == EXIT){
      reward = (switch_status == R_GOOD) ? -1000 : 0;
      return true;
      } else{
      return false;
      }
    */
  }

  /* ================================================
   * Functions related to beliefs and starting states
   * ================================================*/

  double SimpleRockSample::ObsProb(OBS_TYPE obs, const State& state,
                                   ACT_TYPE action) const {
    return 1;
  }

  State* SimpleRockSample::CreateStartState(string type) const {
    //return new SimpleState(8, Random::RANDOM.NextInt(2));
    return new SimpleState(1, 0, 0, R_GOOD, switch_init_pos);
  }

  Belief* SimpleRockSample::InitialBelief(const State* start, string type) const {
    if (type == "DEFAULT" || type == "PARTICLE") {
      vector<State*> particles;

      SimpleState* good_rock = static_cast<SimpleState*>(Allocate(-1, 0.5));
      good_rock->rover_position = 1;
      good_rock->effector_position = 0;
      good_rock->joint1_position = 0;
      good_rock->joint2_position = 0;
      good_rock->switch_position = switch_init_pos;
      good_rock->switch_status = O_GOOD;
      particles.push_back(good_rock);

      SimpleState* bad_rock = static_cast<SimpleState*>(Allocate(-1, 0.5));
      bad_rock->rover_position = 1;
      good_rock->effector_position = 0;
      good_rock->joint1_position = 0;
      good_rock->joint2_position = 0;
      good_rock->switch_position = switch_init_pos;
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
    return 19;
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
    return ValuedAction(A_EAST, -1);
  }

  class SimpleRockSampleEastPolicy: public DefaultPolicy {
  public:
    enum { // action
          A_FLIP = 0, A_EAST = 1, A_WEST = 2, A_CHECK = 3
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
    } else {
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
        << "Joint 1 = " << simple_state.joint1_position << "\n"
        << "Joint 2 = " << simple_state.joint2_position << "\n"
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

    out << "Position belief:"
        << "LEFT" << ":" << pos_probs[0]
        << "MIDDLE" << ":" << pos_probs[1]
        << "EXIT" << ":" << pos_probs[2] << endl;
  }

  void SimpleRockSample::PrintAction(ACT_TYPE action, ostream& out) const {
    if (action == A_FLIP)
      out << "Sample" << endl;
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
