#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
# BIGANN benchmarks of the C++ library on one GPU (device and host vectors, latency, labels, lists):
#   bench/run_bench.sh <BIGANN directory> [certex_bigann binary]
# The BIGANN directory holds base.1B.u8bin, bigann_query.bvecs, and gnd/ (the official ground truth).
D=${1:?usage: run_bench.sh <BIGANN directory> [certex_bigann binary]}
B=${2:-build/certex_bigann}
for m in 59 123; do $B $D 10 $m; $B $D 10 $m 2048 --host; $B $D 10 $m 2048 --latency; done
for m in 59 123; do $B $D 100 $m 2048 --host; done
for L in 10 100 1000; do $B $D 10 123 2048 $L; $B $D 10 123 2048 $L --no-partition; done
$B $D 10 59 2048 --lists=1024; $B $D 10 59 2048 --lists=1024 --host
$B $D 10 59 2048 --lists=1024 --local; $B $D 10 59 2048 --lists=1024 --local --host
$B $D 10 59 2048 --int8; $B $D 10 123 2048 --int8
$B $D 10 0 2048 --tune; $B $D 100 0 2048 --tune --host
$B $D 10 123 2048 --shards=1
