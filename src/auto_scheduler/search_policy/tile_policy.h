/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_scheduler/search_policy/empty_policy.h
 * \brief A simple example of the search policy which always returns the initial naive schedule
 * (state).
 */

#ifndef TVM_AUTO_SCHEDULER_SEARCH_POLICY_TILE_POLICY_H_
#define TVM_AUTO_SCHEDULER_SEARCH_POLICY_TILE_POLICY_H_

#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/measure.h>
#include <tvm/auto_scheduler/cost_model.h>
#include <tvm/auto_scheduler/search_policy.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tile_policy_rules.h"
#include "utils.h"

namespace tvm {
namespace auto_scheduler {

/*! \brief String keys used in parameter map of SketchPolicy. */
struct TileParamKey {
  static constexpr const char* tiling_size = "tiling_size";
  /*! \brief Always allocate this percentage of measurements to random sampled states. */
  static constexpr const char* eps_greedy = "eps_greedy";
  /*! \brief Retry several times if SearchOneRound gets no valid state. */
  static constexpr const char* empty_retry_count = "retry_search_one_round_on_empty";

  struct SampleInitPopulation {
    /*! \brief The minimal size of valid population in the initial sampling. */
    static constexpr const char* min_population = "sample_init_min_population";
    /*! \brief The maximum percentage of measured states in the initial sampling. */
    static constexpr const char* use_measured_ratio = "sample_init_use_measured_ratio";
  };

  struct EvolutionarySearch {
    /*! \brief The population size of evolutionary search. */
    static constexpr const char* population = "evolutionary_search_population";
    /*! \brief The number of iterations performed by generic algorithm.*/
    static constexpr const char* num_iters = "evolutionary_search_num_iters";
    /*! \brief The mutation probability.*/
    static constexpr const char* mutation_prob = "evolutionary_search_mutation_prob";
  };

  struct MultiLevelTiling {
    /*! \brief The structure of multi-level tiling for CPU. */
    static constexpr const char* cpu_structure = "cpu_multi_level_tiling_structure";
    /*! \brief The structure of multi-level tiling for GPU. */
    static constexpr const char* gpu_structure = "gpu_multi_level_tiling_structure";
  };

  /*! \brief The max inner most split factor. */
  static constexpr const char* max_innermost_split_factor = "max_innermost_split_factor";
  /*! \brief The max vectorize size. */
  static constexpr const char* max_vectorize_size = "max_vectorize_size";
  /*! \brief Whether disable compute location changing. */
  static constexpr const char* disable_change_compute_location = "disable_change_compute_location";
};

/*!
 * \brief A simple example of the search policy which always returns the initial naive schedule
 * (state).
 * The key implementation for this structure is `Search()`, check `empty_policy.cc` for more
 * details.
 */
class TilePolicyNode : public SearchPolicyNode {
 public:
  /*! \brief The parameters map for this search policy. */
  Map<String, ObjectRef> params;
  
  /*! \brief Memorize split space for Split. */
  SplitFactorizationMemo split_memo;

  void ParseTilingSizes();

  void print_stages(State state) const;

  int get_target_stage_id();

  int get_iter_id(Stage stage, std::string iter_name);
  
  State Search(int num_measure_trials, int early_stopping, int num_measures_per_round,
               ProgramMeasurer measurer) final;

  std::pair<Array<MeasureInput>, Array<MeasureResult>> ContinueSearchOneRound(
      int num_measure, ProgramMeasurer measurer) final;

  static constexpr const char* _type_key = "auto_scheduler.TilePolicy";
  TVM_DECLARE_FINAL_OBJECT_INFO(TilePolicyNode, SearchPolicyNode);

 private:
  /*!
   * \brief Use a sub function to generate several candidate states in each search round.
   * \returns The generated states
   */
  Array<State> SearchOneRound();

  std::map<int, Array<Optional<Integer>>> tile_size;
};

/*!
 * \brief Managed reference to TilePolicyNode.
 * \sa TilePolicyNode
 */
class TilePolicy : public SearchPolicy {
 public:
  TilePolicy(SearchTask task,
                      Map<String, ObjectRef> params,
                      Optional<Array<SearchCallback>> init_search_callbacks);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TilePolicy, SearchPolicy, TilePolicyNode);
};

}  // namespace auto_scheduler
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_SEARCH_POLICY_EMPTY_POLICY_H_
