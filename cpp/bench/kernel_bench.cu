/*
 * Copyright 2026 Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */

// Throughput of the certified pass: B query rows against N database rows of row_len halves, with
// thresholds that keep about `keep` pairs per query, reported as tensor-core TFLOPS of the fp16
// product.
//
//   kernel_bench <B> <N> <row_len> [keep_per_query]

#include "../src/neighbors/certex/detail/kernels.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <curand.h>

using namespace cuvs::neighbors::certex::detail;

__global__ void to_half(const float* in, half* out, size_t n)
{
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = __float2half(in[i] * 0.125f);
}

int main(int argc, char** argv)
{
  int B       = argc > 1 ? atoi(argv[1]) : 2048;
  long N      = argc > 2 ? atol(argv[2]) : 10000000;
  int K       = argc > 3 ? atoi(argv[3]) : 64;
  double keep = argc > 4 ? atof(argv[4]) : 1000;
  // kernel: automatic, cooperative, pingpong, or sm80
  const std::string kname    = argc > 5 ? argv[5] : "automatic";
  const screen_kernel kernel = kname == "cooperative" ? screen_kernel::cooperative
                               : kname == "pingpong"  ? screen_kernel::pingpong
                               : kname == "sm80"      ? screen_kernel::sm80
                                                      : screen_kernel::automatic;
  half *q, *x;
  float* tmp;
  cudaMalloc(&q, size_t(B) * K * 2);
  cudaMalloc(&x, size_t(N) * K * 2);
  cudaMalloc(&tmp, size_t(N) * K * 4);
  curandGenerator_t g;
  curandCreateGenerator(&g, CURAND_RNG_PSEUDO_DEFAULT);
  curandGenerateNormal(g, tmp, size_t(N) * K, 0.f, 1.f);
  to_half<<<(size_t(N) * K + 255) / 256, 256>>>(tmp, x, size_t(N) * K);
  curandGenerateNormal(g, tmp, size_t(B) * K, 0.f, 1.f);
  to_half<<<(size_t(B) * K + 255) / 256, 256>>>(tmp, q, size_t(B) * K);
  cudaFree(tmp);
  // product of two rows of K halves with entries N(0, 1/64): variance K / 4096; a Gaussian tail
  // keeps `keep` of N
  double sd = std::sqrt(K / 4096.0);
  double z  = 0;  // solve the Gaussian tail N * P(Z > z) = keep by bisection
  for (double lo = 0, hi = 40; hi - lo > 1e-6;) {
    z                                                          = (lo + hi) / 2;
    (0.5 * std::erfc(z / std::sqrt(2.0)) * N > keep ? lo : hi) = z;
  }
  std::vector<float> th(B, float(z * sd));
  float* dth;
  cudaMalloc(&dth, B * 4);
  cudaMemcpy(dth, th.data(), B * 4, cudaMemcpyHostToDevice);
  const int nseg = screen_segments(B, N, K, kernel);
  uint32_t cap   = uint32_t(std::min(4e8, B * keep * 4 + 1e6) / nseg) + 1024;
  int32_t *oq, *ox;
  float* ov;
  uint32_t* cnt;
  cudaMalloc(&oq, cap * 4ull * nseg);
  cudaMalloc(&ox, cap * 4ull * nseg);
  cudaMalloc(&ov, cap * 4ull * nseg);
  cudaMalloc(&cnt, 4ull * nseg);
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  float best = 1e30f;
  uint32_t c = 0;
  for (int rep = 0; rep < 7; ++rep) {
    cudaEventRecord(a);
    screen(q, B, x, N, K, dth, oq, ox, ov, cnt, cap, Filter{}, 0, kernel);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    if (rep) best = std::min(best, ms);
    std::vector<uint32_t> segc(nseg);
    cudaMemcpy(segc.data(), cnt, 4ull * nseg, cudaMemcpyDeviceToHost);
    c = 0;
    for (auto v : segc)
      c += v;
  }
  cudaError_t e = cudaGetLastError();
  std::printf("B=%d N=%ld K=%d: %.3f ms, %.1f TFLOPS, %.0f survivors per query, %s\n",
              B,
              N,
              K,
              best,
              2.0 * B * N * K / (best * 1e-3) / 1e12,
              double(c) / B,
              cudaGetErrorString(e));
  return 0;
}
