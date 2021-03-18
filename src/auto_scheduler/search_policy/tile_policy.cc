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
  State init_state = search_task->compute_dag->init_state;
  int target_stage_id = get_target_stage_id(init_state);
  Stage target = init_state->stages[target_stage_id];

  std::string tile_str = GetStringParam(params, TileParamKey::tiling_size);
  std::cout << "tiling_size: " << tile_str << "\n";
  if (tile_str[tile_str.length() - 1] != ';') {
    tile_str += ';';
  }

  size_t pre = 0, next = tile_str.find(";");
  while (next != std::string::npos) {
    // segment: split info of an iterator, e.g., i@16,8
    std::string segment = tile_str.substr(pre, next - pre);
    if (segment[segment.length() - 1] != ',') {
      segment += ',';
    }

    // get the iterator name
    size_t pos_l = 0, pos_r = segment.find("@");
    std::string iter_name = segment.substr(pos_l, pos_r - pos_l);
    tile_size[iter_name] = Array<Optional<Integer>>();

    // get split factors
    pos_l = pos_r + 1;
    pos_r = segment.find(",");
    while (pos_r != std::string::npos) {
      int size = std::stoi(segment.substr(pos_l, pos_r - pos_l));
      tile_size[iter_name].push_back(Integer(size));
      pos_l = pos_r + 1;
      pos_r = segment.find(",", pos_l);
    }

    pre = next + 1;
    next = tile_str.find(";", pre);
  }
}

int TilePolicyNode::get_target_stage_id(State state) {
  Array<Stage> stages = state->stages;
  for (int i = 0; i < stages.size(); i++) {
    if (stages[i]->op_type == StageKind::kCompute) {
      std::cout << "get_target_stage_id: " << i << "\n";
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

void TilePolicyNode::print_stages(std::string line, State state) const {
  std::cout << "-------" << line << "-------\n";
  printf("state has %ld stages\n", state->stages.size());
  Array<Stage> stages = state->stages;
  for (size_t i = 0; i < stages.size(); i++) {
    Stage cur_stage = stages[i];
    printf("stage %ld, name: %s compute_at: %d\n",
            i, cur_stage->op->name.c_str(), (int)cur_stage->compute_at);
    for (size_t j = 0; j < cur_stage->iters.size(); j++) {
      Iterator cur_iter = cur_stage->iters[j];
      printf("iterator %ld, name: %s kind: %d\n", j, cur_iter->name.c_str(), (int)cur_iter->iter_kind);
    }
  }
}

size_t TilePolicyNode::get_thread_iter(State state, int stage_id) const {  
  Array<Iterator> iters = state->stages[stage_id]->iters;
  for (size_t i = 0; i < iters.size(); i++) {
    std::cout << "stage: " << stage_id << " i: " << i << " annotation: " << (int)iters[i]->annotation << "\n";
    if (iters[i]->annotation == IteratorAnnotation::kThreadX) {
      return i;
    }
  }
  return 0;
}

State TilePolicyNode::Search(int num_measure_trials, int early_stopping,
                              int num_measures_per_round, ProgramMeasurer measurer) {
    State init_state = search_task->compute_dag->init_state;
    int target_stage_id = get_target_stage_id(init_state);
    measurer->Reset();
    Array<MeasureInput> inputs;
    Array<MeasureResult> results;

    // 1. Get tiling sizes
    ParseTilingSizes();

    // Add read cache stages (shared memory)
    if (need_smem_tiling) {
      print_stages("add read cache", init_state);

      const std::set<int>& producers = GetProducers(search_task, init_state, target_stage_id);
      int stage_offset = 0;
      for (int producer : producers) {
        init_state.cache_read(producer + stage_offset, "shared", {target_stage_id}, search_task->compute_dag);
        stage_offset++;
        target_stage_id++;
      }
    } else {
      std::cout << "-------No Shared Memory Tiling-------\n";
    }

    // Add write cache stages (register)
    if (need_reg_tiling) {
      print_stages("add write cache", init_state);

      std::vector<int> read_cache_stages = GetCacheReadStages(init_state);
      read_cache_stages.push_back(target_stage_id);
      int stage_offset = 0;
      for (int i = 0; i < read_cache_stages.size(); i++) {
        std::cout << "calling cache_write i: " << read_cache_stages[i] << " offset: " << stage_offset << "\n";
        init_state.cache_write(read_cache_stages[i] + stage_offset, "local", search_task->compute_dag);
        std::cout << "cache_write finished\n";
        stage_offset++;
        target_stage_id++;
        print_stages(std::to_string(i), init_state);
      }
      
      // int added_stage_id = target_stage_id++; // added before the target stage
      // print_stages(init_state);
      // size_t thrd_iter_idx = get_thread_iter(init_state, target_stage_id);
      // std::cout << "thrd_iter_idx " << thrd_iter_idx << "\n";
      // init_state.compute_at(added_stage_id, target_stage_id, init_state->stages[target_stage_id]->iters[thrd_iter_idx]);
    } else {
      std::cout << "-------No Register Tiling-------";
    }

    // Split axises (spatial only)
    Stage target_stage = init_state->stages[target_stage_id];
    for (auto it : target_stage->iters) {
      if (it->iter_kind == IteratorKind::kSpatial) {
        init_state.split(target_stage_id, it, tile_size[it->name]);
      }
    }
    std::cout << "[2. Split]\n" << init_state.ToStr() << "\n";

    // Reorder axis according to the memory layers
    print_stages("reoder axes", init_state);
    target_stage = init_state->stages[target_stage_id];
    Array<Iterator> block_axis, thread_axis, space_axis, reduce_axis, after;
    std::cout << "target_cache_stage: " << target_stage 
              << " iters: " << target_stage->iters << "\n";
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
    for (int i = 0; i < space_axis.size(); i++)  after.push_back(space_axis[i]);
    for (int i = 0; i < reduce_axis.size(); i++) after.push_back(reduce_axis[i]);

    init_state.reorder(target_stage_id, after);
    std::cout << "[3. Reorder]\n" << init_state.ToStr() << "\n";

    // Fuse and bind axises to threads and blocks
    print_stages("fuse axes", init_state);
    Iterator bx = init_state.fuse(target_stage_id, block_axis);
    Iterator tx = init_state.fuse(target_stage_id, thread_axis);
    print_stages("bind axes", init_state);
    target_stage = init_state->stages[target_stage_id];
    Iterator new_bx = init_state.bind(target_stage_id, bx, IteratorAnnotation::kBlockX);
    Iterator new_tx = init_state.bind(target_stage_id, tx, IteratorAnnotation::kThreadX);
    std::cout << "[4.Fuse]\n" << init_state.ToStr() << "\n";
    
    // Step
    int target_cache_stage_id = GetTargetCacheStageId(init_state, target_stage_id);
    if (target_cache_stage_id != -1) {
      init_state.compute_at(target_cache_stage_id, target_stage_id, new_tx);
      Stage target_cache_stage = init_state->stages[target_cache_stage_id];

      after.clear();
      // for (auto it : target_cache_stage->iters) {
      //   if (it->iter_kind == IteratorKind::kReduction) {
      //     init_state.split(target_cache_stage_id, it, tile_size[it->name]);
      //   }
      // }

      // target_cache_stage = init_state->stages[target_cache_stage_id];
      reduce_axis = GetIteratorsByKind(target_cache_stage, IteratorKind::kReduction);
      space_axis = GetIteratorsByKind(target_cache_stage, IteratorKind::kSpatial);
      for (size_t i = 0; i < reduce_axis.size(); i++) after.push_back(reduce_axis[i]);
      for (size_t i = 0; i < space_axis.size(); i++) after.push_back(space_axis[i]);
      
      init_state.reorder(target_cache_stage_id, after);
      print_stages("Step", init_state);
      std::cout << "[Step]\n" << init_state.ToStr() << "\n";
    }

    // Cooperative fetching
    for (int read_cache_stage_id : GetReadCacheStages(init_state)) {
      if (target_cache_stage_id != -1) {
        Iterator outer_reduce_iter = init_state->stages[target_cache_stage_id]->iters[0];
        init_state.compute_at(read_cache_stage_id, target_cache_stage_id, outer_reduce_iter);
      } else {
        init_state.compute_at(read_cache_stage_id, target_stage_id, new_tx);
      }
      // Stage read_cache_stage = init_state->stages[read_cache_stage_id];
      // for (auto it : target_stage->iters) {
      //   if (it->iter_kind == IteratorKind::kSpatial) {
      //     init_state.split(read_cache_stage_id, it, tile_size[it->name][1]);
      //   }
      // }
    }

    // Register tile
    for (int write_cache_stage_id : GetWriteCacheStages(init_state)) {
      if (write_cache_stage_id != target_cache_stage_id) {
        Stage target_cache_stage = init_state->stages[target_cache_stage_id];
        int idx = 0;
        while (target_cache_stage->iters[idx]->iter_kind == IteratorKind::kReduction) {
          idx++;
        }
        Iterator inner_reduce_iter = target_cache_stage->iters[idx - 1];
        init_state.compute_at(write_cache_stage_id, target_cache_stage_id, inner_reduce_iter);
      }
    }
    print_stages("Register tile", init_state);
    std::cout << "[Register tile]\n" << init_state.ToStr() << "\n";

    // Measure
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
