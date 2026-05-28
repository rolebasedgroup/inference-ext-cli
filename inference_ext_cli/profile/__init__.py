"""SLA profiling pipeline for RBG inference workloads.

This package implements the full profiling workflow:
1. Deploy temporary RBG instances with various GPU configurations
2. Run AIPerf benchmarks to measure TTFT/ITL/throughput
3. Sweep parallelization mappings (TP/TEP/DEP) to find optimal configs
4. Perform interpolation sweeps for prefill (ISL) and decode (ISL x concurrency)
5. Generate Kubernetes ConfigMap with profiling data for the RBG planner
"""
