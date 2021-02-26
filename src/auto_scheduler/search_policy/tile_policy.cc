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
 * \file auto_scheduler/search_policy/empty_policy.cc
 * \brief A simple example of the search policy which always returns the initial naive schedule
 * (state).
 */

#include "tile_policy.h"

#include <tvm/auto_scheduler/measure.h>
#include <tvm/runtime/registry.h>

#include <utility>

#include "utils.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(TilePolicyNode);

TilePolicy::TilePolicy(SearchTask task, Optional<Array<SearchCallback>> init_search_callbacks) {
  auto node = make_object<TilePolicyNode>();
  node->search_task = task;

  // Run init_search_callbacks before the search process
  // This Interface is usually used to set some init status
  if (init_search_callbacks) {
    node->RunCallbacks(init_search_callbacks.value());
  }

  data_ = std::move(node);
}

State TilePolicyNode::Search(int num_measure_trials, int early_stopping,
                              int num_measures_per_round, ProgramMeasurer measurer) {
  // Basic design principe: `SearchOneRound()` several times to get candidate states,
  // measure them and return the best one
  // Measure is disabled if num_measure_trials <= 1
  if (num_measure_trials <= 1) {
    const auto& res = SearchOneRound();
    ICHECK_GT(res.size(), 0);

    return res[0];
  } else {
    Array<MeasureInput> inputs;
    Array<MeasureResult> results;

    measurer->Reset();
    int ct = 0;
    // In each round, we call SearchOneRound to get several candidate states,
    // then use ProgramMeasurer to measure their performance.
    while (ct < num_measure_trials) {
      const auto& res = SearchOneRound();
      ct += res.size();
      // Build MeasureInputs for measuring
      inputs.clear();
      for (const auto& state : res) {
        inputs.push_back(MeasureInput(search_task, state));
      }
      // Perform measurement.
      // ProgramMeasurer will record the state with best performance during measure process
      results = measurer->Measure(search_task, GetRef<SearchPolicy>(this), inputs);
    }

    // Return a state with best measured performance
    return measurer->best_state[search_task->workload_key];
  }
}

std::pair<Array<MeasureInput>, Array<MeasureResult>> TilePolicyNode::ContinueSearchOneRound(
    int num_measure, ProgramMeasurer measurer) {
  Array<State> best_states;
  Array<MeasureInput> inputs;
  Array<MeasureResult> results;

  // Search one round to get promising states
  PrintTitle("Search", verbose);
  best_states = SearchOneRound();

  // Measure these states
  PrintTitle("Measure", verbose);
  for (const auto& state : best_states) {
    inputs.push_back(MeasureInput(search_task, state));
  }
  results = measurer->Measure(search_task, GetRef<SearchPolicy>(this), inputs);

  return std::make_pair(std::move(inputs), std::move(results));
}

// As an example policy, TilePolicy always returns a init state
Array<State> TilePolicyNode::SearchOneRound() {
  Array<State> res;

  // LHY
  printf("init_state has %ld stages\n", search_task->compute_dag->init_state->stages.size());
  Array<Stage> stages = search_task->compute_dag->init_state->stages;
  for (size_t i = 0; i < stages.size(); i++) {
    Stage cur_stage = stages[i];
    printf("stage %ld, name: %s compute_at: %d\n",
            i, cur_stage->op->name.c_str(), (int)cur_stage->compute_at);
    for (size_t j = 0; j < cur_stage->iters.size(); j++) {
      Iterator cur_iter = cur_stage->iters[j];
      printf("iterator %ld, name: %s\n", j, cur_iter->name.c_str());
    }
  }
  
  // Simply return the initial naive schedule (state).
  res.push_back(search_task->compute_dag->init_state);

  return res;
}

TVM_REGISTER_GLOBAL("auto_scheduler.TilePolicy")
    .set_body_typed([](SearchTask task, Optional<Array<SearchCallback>> init_search_callbacks) {
      return TilePolicy(task, init_search_callbacks);
    });

}  // namespace auto_scheduler
}  // namespace tvm
