/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <omp.h>
#include "params.h"
#include <omp.h> 

float sigmoid(const float x)
{
  return 1.0f / (1.0f + expf(-x));
}

void postprocess(const float *cls_input,float *box_input,const float *dir_cls_input,const float *anchors,const float *anchor_bottom_heights,float *bndbox_output,int *object_counter,const float min_x_range,const float max_x_range,const float min_y_range,const float max_y_range,const int feature_x_size,const int feature_y_size,const int num_anchors,const int num_classes,const int num_box_values,const float score_thresh,const float dir_offset)
{
  int loc_index = omp_get_team_num();
  int itanchor = omp_get_thread_num();
  if (itanchor >= num_anchors) 
    return ;
  int col = loc_index % feature_x_size;
  int row = loc_index / feature_x_size;
  float x_offset = min_x_range + col * (max_x_range - min_x_range) / (feature_x_size - 1);
  float y_offset = min_y_range + row * (max_y_range - min_y_range) / (feature_y_size - 1);
  int cls_offset = loc_index * num_anchors * num_classes + itanchor * num_classes;
  float dev_cls[2] = {(- 1.f), (0.f)};
  const float *scores = cls_input + cls_offset;
  float max_score = sigmoid(scores[0]);
  int cls_id = 0;
  for (int i = 1; i <= num_classes - 1; i += 1) {
    float cls_score = sigmoid(scores[i]);
    if (cls_score > max_score) {
      max_score = cls_score;
      cls_id = i;
    }
  }
  dev_cls[0] = (static_cast < float  >  (cls_id));
  dev_cls[1] = max_score;
  if (dev_cls[1] >= score_thresh) {
    int box_offset = loc_index * num_anchors * num_box_values + itanchor * num_box_values;
    int dir_cls_offset = loc_index * num_anchors * 2 + itanchor * 2;
    const float *anchor_ptr = anchors + itanchor * 4;
    float z_offset = anchor_ptr[2] / 2 + anchor_bottom_heights[itanchor / 2];
    float anchor[7] = {x_offset, y_offset, z_offset, anchor_ptr[0], anchor_ptr[1], anchor_ptr[2], anchor_ptr[3]};
    float *box_encodings = box_input + box_offset;
    float xa = anchor[0];
    float ya = anchor[1];
    float za = anchor[2];
    float dxa = anchor[3];
    float dya = anchor[4];
    float dza = anchor[5];
    float ra = anchor[6];
    float diagonal = sqrtf(dxa * dxa + dya * dya);
    box_encodings[0] = box_encodings[0] * diagonal + xa;
    box_encodings[1] = box_encodings[1] * diagonal + ya;
    box_encodings[2] = box_encodings[2] * dza + za;
    box_encodings[3] = expf(box_encodings[3]) * dxa;
    box_encodings[4] = expf(box_encodings[4]) * dya;
    box_encodings[5] = expf(box_encodings[5]) * dza;
    box_encodings[6] = box_encodings[6] + ra;
    float yaw;
    int dir_label = dir_cls_input[dir_cls_offset] > dir_cls_input[dir_cls_offset + 1]?0 : 1;
    const float period = (float )3.14159265358979323846;
    float val = box_input[box_offset + 6] - dir_offset;
    float dir_rot = val - floorf(val / (period + 1e-8f)) * period;
    yaw = dir_rot + dir_offset + period * dir_label;
    int resCount;
{
      resCount = object_counter[0];
      object_counter[0]++;
    }
    bndbox_output[0] = (resCount + 1);
    float *data = bndbox_output + 1 + resCount * 9;
    data[0] = box_input[box_offset];
    data[1] = box_input[box_offset + 1];
    data[2] = box_input[box_offset + 2];
    data[3] = box_input[box_offset + 3];
    data[4] = box_input[box_offset + 4];
    data[5] = box_input[box_offset + 5];
    data[6] = yaw;
    data[7] = dev_cls[0];
    data[8] = dev_cls[1];
  }
}

int main(int argc,char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n",argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  class Params p;
// constant values defined in params.h
  const float min_x_range = p . min_x_range;
  const float max_x_range = p . max_x_range;
  const float min_y_range = p . min_y_range;
  const float max_y_range = p . max_y_range;
  const int feature_x_size = p . feature_x_size;
  const int feature_y_size = p . feature_y_size;
  const int num_anchors = Params::num_anchors;
  const int num_classes = Params::num_classes;
  const int num_box_values = p . num_box_values;
  const float score_thresh = p . score_thresh;
  const float dir_offset = p . dir_offset;
  const int len_per_anchor = Params::len_per_anchor;
  const int num_dir_bins = p . num_dir_bins;
  const int feature_size = feature_x_size * feature_y_size;
  const int feature_anchor_size = feature_size * num_anchors;
  const int cls_size = feature_anchor_size * num_classes;
  const int box_size = feature_anchor_size * num_box_values;
  const int dir_cls_size = feature_anchor_size * num_dir_bins;
  const int bndbox_size = feature_anchor_size * 9 + 1;
  const int cls_size_byte = (cls_size * sizeof(float ));
  const int box_size_byte = (box_size * sizeof(float ));
  const int dir_cls_size_byte = (dir_cls_size * sizeof(float ));
  const int bndbox_size_byte = (bndbox_size * sizeof(float ));
// input of the post-process kernel
  float *cls_input = (float *)(malloc(cls_size_byte));
  float *box_input = (float *)(malloc(box_size_byte));
  float *dir_cls_input = (float *)(malloc(dir_cls_size_byte));
// output of the post-process kernel
  float *bndbox_output = (float *)(malloc(bndbox_size_byte));
  const float *anchors = p . anchors;
  const float *anchor_bottom_heights = p . anchor_bottom_heights;
  int object_counter[1];
// random values
  srand(123);
  for (int i = 0; i <= cls_size - 1; i += 1) {
    cls_input[i] = (rand()) / ((float )2147483647);
  }
  for (int i = 0; i <= box_size - 1; i += 1) {
    box_input[i] = (rand()) / ((float )2147483647);
  }
  for (int i = 0; i <= dir_cls_size - 1; i += 1) {
    dir_cls_input[i] = (rand()) / ((float )2147483647);
  }
{
    double time = 0.0;
    for (int i = 0; i <= repeat - 1; i += 1) {
      object_counter[0] = 0;
      auto start = std::chrono::_V2::steady_clock::now();
{
{
          postprocess(cls_input,box_input,dir_cls_input,anchors,anchor_bottom_heights,bndbox_output,object_counter,min_x_range,max_x_range,min_y_range,max_y_range,feature_x_size,feature_y_size,num_anchors,num_classes,num_box_values,score_thresh,dir_offset);
        }
      }
      auto end = std::chrono::_V2::steady_clock::now();
      time += (std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count());
    }
    printf("Average execution time of postprocess kernel: %f (us)\n",time * 1e-3f / repeat);
  }
  double checksum = 0.0;
  
#pragma omp parallel for reduction (+:checksum)
  for (int i = 0; i <= bndbox_size - 1; i += 1) {
    checksum += bndbox_output[i];
  }
  printf("checksum = %lf\n",checksum / bndbox_size);
  free(cls_input);
  free(box_input);
  free(dir_cls_input);
  free(bndbox_output);
  return 0;
}
