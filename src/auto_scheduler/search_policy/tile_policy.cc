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

TilePolicy::TilePolicy(SearchTask task,
                      Map<String, ObjectRef> params,
                      Optional<Array<SearchCallback>> init_search_callbacks) {
  auto node = make_object<TilePolicyNode>();
  node->search_task = task;
  node->params = std::move(params);

  // Run init_search_callbacks before the search process
  // This Interface is usually used to set some init status
  if (init_search_callbacks) {
    node->RunCallbacks(init_search_callbacks.value());
  }

  data_ = std::move(node);
}

void TilePolicyNode::ParseTilingSizes() {
  int target_stage_id = get_target_stage_id();
  Stage target = search_task->compute_dag->init_state->stages[target_stage_id];

  std::string tile_str = GetStringParam(params, TileParamKey::tiling_size);
  std::cout << "tiling_size: " << tile_str << "\n";
  if (tile_str[tile_str.length() - 1] != ';') {
    tile_str += ';';
  }

  size_t pre = 0, next = tile_str.find(";");
  while (next != std::string::npos) {
    std::string segment = tile_str.substr(pre, next - pre);
    if (segment[segment.length() - 1] != ',') {
      segment += ',';
    }

    size_t pos_l = 0, pos_r = segment.find("@");
    std::string iter_name = segment.substr(pos_l, pos_r - pos_l);
    int iter_id = get_iter_id(target, iter_name);
    tile_size[iter_id] = Array<Optional<Integer>>();

    pos_l = pos_r + 1;
    pos_r = segment.find(",");
    while (pos_r != std::string::npos) {
      int size = std::stoi(segment.substr(pos_l, pos_r - pos_l));
      tile_size[iter_id].push_back(Integer(size));
      pos_l = pos_r + 1;
      pos_r = segment.find(",", pos_l);
    }    

    pre = next + 1;
    next = tile_str.find(";", pre);
  }
}

int TilePolicyNode::get_target_stage_id() {
  Array<Stage> stages = search_task->compute_dag->init_state->stages;
  for (int i = 0; i < stages.size(); i++) {
    if (stages[i]->op_type == StageKind::kCompute) {
      return i;
    }
  }
  return -1;
}

int TilePolicyNode::get_iter_id(Stage stage, std::string iter_name) {
  for (int i = 0; i < stage->iters.size(); i++) {
    if (stage->iters[i]->name.compare(iter_name) == 0) {
      return i;
    }
  }
  return -1;
}

void TilePolicyNode::print_stages(State state) const {
  printf("state has %ld stages\n", state->stages.size());
  Array<Stage> stages = state->stages;
  for (size_t i = 0; i < stages.size(); i++) {
    Stage cur_stage = stages[i];
    printf("stage %ld, name: %s compute_at: %d\n",
            i, cur_stage->op->name.c_str(), (int)cur_stage->compute_at);
    for (size_t j = 0; j < cur_stage->iters.size(); j++) {
      Iterator cur_iter = cur_stage->iters[j];
      printf("iterator %ld, name: %s\n", j, cur_iter->name.c_str());
    }
  }
}

State TilePolicyNode::Search(int num_measure_trials, int early_stopping,
                              int num_measures_per_round, ProgramMeasurer measurer) {
    State init_state = search_task->compute_dag->init_state;
    measurer->Reset();
    Array<MeasureInput> inputs;
    Array<MeasureResult> results;

    // 1. Get tiling sizes
    ParseTilingSizes();

    // 2. Split axises
    int target_stage_id = get_target_stage_id();
    Stage target_stage = init_state->stages[target_stage_id];
    for (auto it : tile_size) {
      int iter_id = it.first;
      Array<Optional<Integer>> lengths = it.second;
      Iterator iter = target_stage->iters[it.first];
      bool inner_to_outer = true;
      init_state.split(target_stage_id, iter, lengths);
    }
    std::cout << "[2. Split]\n" << init_state.ToStr() << "\n";

    // 3. Reorder axis according to the memory layers
    print_stages(init_state);
    target_stage_id = get_target_stage_id();
    target_stage = init_state->stages[target_stage_id];
    Array<Iterator> block_axis, thread_axis, space_axis, reduce_axis, after;
    for (int i = 0; i < target_stage->iters.size(); i++) {
      Iterator it = target_stage->iters[i];
      if (it->iter_kind == IteratorKind::kSpatial) {
        if (StrEndsWith(it->name, ".0")) {
          block_axis.push_back(it);
        } else if (StrEndsWith(it->name, ".1")) {
          thread_axis.push_back(it);
        } else if (StrEndsWith(it->name, ".2")) {
          space_axis.push_back(it);
        } else {
          // Should not reach here
          ICHECK(0);
        }
      } else if (it->iter_kind == IteratorKind::kReduction) {
        reduce_axis.push_back(it);
      } else {
        // Should not reach here
        ICHECK(0);
      }
    }
    ICHECK(block_axis.size() == thread_axis.size());
    ICHECK(thread_axis.size() == space_axis.size());

    for (int i = 0; i < block_axis.size(); i++)  after.push_back(block_axis[i]);
    for (int i = 0; i < thread_axis.size(); i++) after.push_back(thread_axis[i]);
    for (int i = 0; i < reduce_axis.size(); i++) after.push_back(reduce_axis[i]);
    for (int i = 0; i < space_axis.size(); i++)  after.push_back(space_axis[i]);

    init_state.reorder(target_stage_id, after);
    std::cout << "[3. Reorder]\n" << init_state.ToStr() << "\n";

    // 4. Fuse and bind axises to threads and blocks
    print_stages(init_state);
    Iterator bx = init_state.fuse(target_stage_id, block_axis);
    Iterator tx = init_state.fuse(target_stage_id, thread_axis);
    print_stages(init_state);
    target_stage_id = get_target_stage_id();
    target_stage = init_state->stages[target_stage_id];
    init_state.bind(target_stage_id, bx, IteratorAnnotation::kBlockX);
    init_state.bind(target_stage_id, tx, IteratorAnnotation::kThreadX);
    std::cout << "[4.Fuse]\n" << init_state.ToStr() << "\n";

    // 5. Add read cache stages (shared memory)
    print_stages(init_state);
    RuleAddCacheRead rule;
    const std::set<int>& producers = GetProducers(search_task, init_state, target_stage_id);
    int stage_offset = 0;
    for (int producer : producers) {
      auto res = rule.Apply(*this, init_state, producer + stage_offset);
      stage_offset++;
      init_state = res[0].first;
    }

    // 6. Add write cache stages (register)


    // 7. Measure
    inputs.push_back(MeasureInput(search_task, init_state));
    results = measurer->Measure(search_task, GetRef<SearchPolicy>(this), inputs);

    return init_state;
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
  State init_state = search_task->compute_dag->init_state;

  int stage_id = 2;

  RuleMultiLevelTiling mlt_rule;
  TileGenerationRule::ConditionKind ckind = mlt_rule.MeetCondition(*this, init_state, stage_id);
  if (ckind == TileGenerationRule::ConditionKind::kApplyAndSkipRest) {
    printf("ConditionKind::kApplyAndSkipRest\n");
  }

  if (ckind == TileGenerationRule::ConditionKind::kSkip) {
    printf("ConditionKind::kSkip\n");
  }

  State tmp_state = mlt_rule.Apply(*this, init_state, stage_id)[0].first;
  printf("Applied TileGenerationRule\n%s\n", tmp_state.ToStr().c_str());

  InitFillTileSize ifts_rule;
  std::mt19937 rand;
  PopulationGenerationRule::ResultKind rkind = ifts_rule.Apply(this, &tmp_state, &rand);
  printf("Applied InitFillTileSize %d\n%s\n", (int)rkind, tmp_state.ToStr().c_str());

  // tmp_state = search_task->compute_dag.InferBound(tmp_state);
  // printf("Infer Bound\n%s\n", tmp_state.ToStr().c_str());

  InitThreadBind itb_rule;
  rkind = itb_rule.Apply(this, &tmp_state, &rand);
  printf("Applied InitThreadBind %d\n%s\n", (int)rkind, tmp_state.ToStr().c_str());
  
  // Simply return the initial naive schedule (state).
  res.push_back(tmp_state);

  return res;
}

TVM_REGISTER_GLOBAL("auto_scheduler.TilePolicy")
    .set_body_typed([](SearchTask task, Map<String, ObjectRef> params, Optional<Array<SearchCallback>> init_search_callbacks) {
      return TilePolicy(task, params, init_search_callbacks);
    });

}  // namespace auto_scheduler
}  // namespace tvm
