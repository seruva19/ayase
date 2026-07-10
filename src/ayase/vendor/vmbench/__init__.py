"""Vendored VMBench scoring utilities.

VMBench — "A Benchmark for Perception-Aligned Video Motion Generation"
(AMAP-ML, ICCV 2025, arXiv:2503.10076). Upstream: github.com/AMAP-ML/VMBench,
licensed Apache-2.0. Only the pure-NumPy scoring functions are vendored here;
the heavy detection/pose backends are replaced by the toolkit's own rtmlib
RTMPose backend, so no mmpose/mmcv/CUDA build is required.
"""
