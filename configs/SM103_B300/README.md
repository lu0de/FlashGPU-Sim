# SM103_B300 Configuration

PTX execution-driven configuration for the NVIDIA B300 (Blackwell Ultra,
`sm_103a`). It starts from [`SM100_B200`](../SM100_B200/README.md) and replaces
the parameters that were measured to differ on B300 hardware. Everything not
listed below is inherited from `SM100_B200`, including
`config_ampere_islip.icnt` and `sass_primary_hints.rules`, which are copies.

The compute capability is kept at 10.0 so that `sm_100a` PTX loads unchanged;
the device itself reports 10.3.

## Method

- Device facts come from `cudaDeviceGetAttribute` and `nvidia-smi` on a
  B300 SXM6 (148 SMs, 32 resident blocks per SM, 228 / 227 KiB shared memory,
  126 MiB L2, 7680-bit memory bus).
- Hardware timing comes from single-purpose microbenchmarks profiled with
  Nsight Compute 2026.1.1 (CUDA 13.2), built for `sm_103a`.
- The same kernels, built for `sm_100a`, run in the simulator; each parameter
  was adjusted until the simulated value matched the hardware value.
- Clocks could not be locked on the measurement platform, so every hardware
  cycle count was taken at the boost workpoint: SM 2032 MHz, L2 1981 MHz
  (2032 x the measured L2/SM cycle ratio 1.96 / 2.01), DRAM 3996 MHz. The
  configuration models that workpoint.

This is a per-parameter calibration against microbenchmarks. It is not an
end-to-end validation of complete kernels.

## Parameters changed relative to SM100_B200

Hardware and simulated values are in core cycles at 2032 MHz unless a unit is
given.

| Option | B200 | B300 | Hardware | Simulated | Basis |
| --- | --- | --- | --- | --- | --- |
| `-gpgpu_clock_domains` | 1080:1080:1155:3996 | 2032:2032:1981:3996 | SM 2.03 GHz, L2 1.98 GHz | - | Measured workpoint |
| `-gpgpu_shader_cta` | 24 | 32 | 32 | - | Device attribute |
| `-gpgpu_kernel_launch_latency` | 5000 | 6422 | 6434 | 6434 | Empty kernel, grid 148, block 512 |
| `-ptx_opcode_latency_int` (ADD) | 4 | 5 | 5.02 | 5.00 | Dependent add chain, see limit 1 |
| `-ptx_opcode_initiation_int` (MUL, MAD) | 1, 1 | 2, 2 | 64 ops/cycle/SM | - | 128 / 64 |
| `-ptx_opcode_latency_fp` (MIN/MAX) | 5 | 6 | 10.04 for min + add | 10.01 | Pair minus the 4.04-cycle add |
| `-ptx_opcode_initiation_dp` | 64 | 76 | 1.684 FMA/cycle/SM | - | 128 / 1.684 |
| `-ptx_opcode_latency_sfu` | 28 | 20 | 24.04 for rcp + add | 24.01 | Pair minus the 4-cycle add |
| `-ptx_opcode_latency_ex2` | 18 | 14 | 14.04 | 14 | Dependent ex2 chain |
| `-ptx_opcode_initiation_ex2` | inherits SFU (8) | 4 | 31.99 ops/cycle/SM | - | Other SFU ops stay at 16 ops/cycle/SM |
| `-ptx_opcode_latency_tensor` (first entry) | 34 | 20 | 20.08 | - | `mma.sync` m16n8k16 fp16 |
| `-ptx_opcode_tcgen05_mma_completion_tail_latency` | 160 | 238 | 243.2 | 243.1 | MMA to commit to mbarrier, N = 256, see limit 3 |
| `-ptx_opcode_tcgen05_st_completion_latency` | default | 10,10,10,13,16,23,38,70 | - | - | Hardware-derived, x1 to x128 |
| `-ptx_opcode_tcgen05_ld_completion_latency` | default | 29,29,30,32,36,44,60,92 | - | - | Hardware-derived, x1 to x128 |
| `-ptx_opcode_tcgen05_ld_initiation_interval` | default | 1 for every width | 0.2 to 0.68 cycles/op | - | Simulator floor is 1 |
| `-gpgpu_smem_latency` | 19 | 15 | 29.0 | 29.0 | Shared-memory load chain |
| `-gpgpu_l1_latency` | 39 | 1 | 39.5 | 58.4 | Simulator floor, see limit 4 |
| `-gpgpu_l2_rop_latency` | 260 | 286 | 285.4 | 284.2 | L2 hit |
| `-dram_latency` | 520 | 666 | 659.2 | 665.6 | Cold DRAM miss |
| `-gpgpu_simple_dram_service_rate_num` / `_den` | 15 / 4 | 7 / 2 | 6.99 TB/s | 7.16 TB/s nominal | See limit 2 |
| `-gpgpu_simple_dram_max_inflight` | 7220 | 4590 | - | 4585 observed | 666 / 2032 x 3996 x 3.5 |
| `-gpgpu_dram_expected_bandwidth_TBp` | 7.7 | 7.16 | - | - | Follows the service rate |
| `-gpgpu_l2_expected_bandwidth_TBp` | 11.8 | 20.28 | 19.62 TB/s | 20.13 TB/s | 5/3 x 32 B x 192 x 1981 MHz |
| `-gpgpu_tma_max_inflight` | 3200 | 2728 | 682.5 | - | 4 x cold dependent-load latency |
| `-gpgpu_cta_replacement_latency` | 1200 | 1187 | 3726 | 3726 | Gap plus active time of a TMEM kernel, see limit 5 |
| `-gpgpu_mbarrier_arrive_latency` | 29 | 37 | 53.0 | 53.5 | Self arrive |
| `-gpgpu_mbarrier_phase_wakeup_latency` | 110 | 72 | 78.4 | see limit 6 | Ping-pong, one way |
| `-gpgpu_named_barrier_arrive_visibility_latency` | 24 | 0 | 40.1 | 46.0 | Ping-pong, one way; already at the floor |
| `-gpgpu_barrier_release_latency` | default (0) | 40 | 50.5 | 50.5 | `bar.sync` with 512 threads |

`-ptx_opcode_initiation_ex2` is a new option. It sets the initiation interval
of EX2 separately from the other SFU operations; 0 (the default) inherits
`-ptx_opcode_initiation_sfu`.

## Measured and found equal to SM100_B200

FP32 ADD / MUL / MAD latency 4, integer logic 4 with interval 2, SETP / SELP
5, packed `cvt.f16x2.f32` 4 with interval 2, packed `f32x2` interval 2, DP
latency 64, TCGen05 dense FP16 rate 8192 FLOP/SM-cycle, and the L2 data-port
ratio 5/3.

## Known limits

1. The integer ADD latency of 5 is a PTX-level value. The measured chain
   alternates two SASS instructions per PTX add, so a single SASS add may be 4.
2. End-to-end global-load bandwidth saturates near 5.11 TB/s in the simulator
   while hardware reaches 6.99 TB/s. The DRAM model alone reaches its nominal
   7.16 TB/s, so the cap is on the SM and interconnect path. TMA loads are
   closer: 125.9 B/cycle simulated against 119.8 B/cycle measured.
3. The idle commit-to-mbarrier overhead is 13 cycles in the simulator and 106
   on hardware. The difference is folded into the TCGen05 completion tail so
   that the total matches. The per-MMA floor of about 50 cycles seen on
   hardware at N = 64 is not modeled.
4. The simulator has a fixed L1 pipeline overhead of about 58 cycles, above
   the measured 39.5, so `-gpgpu_l1_latency` is at its minimum.
5. `-gpgpu_cta_replacement_latency` absorbs TMEM allocation and release, which
   are not modeled separately. Kernels that do not use TMEM are over-penalized:
   their CTA gap is 1541 cycles simulated against 536 measured.
6. The simulated response to `-gpgpu_mbarrier_phase_wakeup_latency` is not
   continuous (36.7 cycles at 71, 84.5 at 72) and interacts with
   `-gpgpu_mbarrier_trywait_latency`. The value 72 is the closest setting; the
   interaction is unresolved.
7. Block-scaled MMA kinds (NVFP4, FP8), INT8 tensor operations, and
   `cta_group::2` are outside this configuration.
