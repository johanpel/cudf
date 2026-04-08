/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/types.hpp>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace io {

/**
 * @brief CUPTI-based implementation of kernel_timing_observer.
 *
 * Uses the CUPTI Activity API to collect GPU kernel execution timestamps.
 * Callback-based correlation maps kernel records to pipeline stages.
 *
 * If CUPTI subscription fails (e.g., another subscriber exists), the observer
 * degrades gracefully: stage_begin/stage_end become no-ops and populate_stats
 * adds no timing entries.
 *
 * For Blackwell+ GPUs (CUDA 12.8+), the Hardware Event System (HES) can be
 * enabled for hardware-level timestamps with lower overhead. HES must be
 * enabled before CUDA context creation, so the observer must be constructed
 * early in that case.
 */
class cupti_timing_observer : public kernel_timing_observer {
 public:
  struct config {
    bool use_hes{true};  ///< Try HES (Blackwell+, CUDA 12.8+). Falls back to
                         ///< SW tracing if unavailable or CUDA context already exists.
  };

  explicit cupti_timing_observer(config cfg = {.use_hes = true});
  ~cupti_timing_observer() override;

  cupti_timing_observer(cupti_timing_observer const&)            = delete;
  cupti_timing_observer& operator=(cupti_timing_observer const&) = delete;
  cupti_timing_observer(cupti_timing_observer&&)                 = delete;
  cupti_timing_observer& operator=(cupti_timing_observer&&)      = delete;

  void stage_begin(parquet_pipeline_stage stage,
                   uint32_t sub_stage_id,
                   rmm::cuda_stream_view stream) override;

  void stage_end(parquet_pipeline_stage stage,
                 uint32_t sub_stage_id,
                 rmm::cuda_stream_view stream) override;

  void populate_stats(parquet_pipeline_stats& stats) override;

  /**
   * @brief Returns whether the observer is actively collecting data.
   *
   * @return true if CUPTI was successfully initialized
   */
  [[nodiscard]] bool is_active() const;

  /**
   * @brief Returns whether HES (hardware timestamps) is active.
   *
   * @return true if HES was requested and successfully enabled
   */
  [[nodiscard]] bool is_hes_active() const;

  /**
   * @brief Returns the CUptiResult from the HES enablement attempt.
   *
   * Useful for diagnosing why HES failed (e.g. 16=NOT_COMPATIBLE,
   * 35=INSUFFICIENT_PRIVILEGES).
   *
   * @return 0 (CUPTI_SUCCESS) if HES enabled, or the CUptiResult error code
   */
  [[nodiscard]] int hes_error_code() const;

  /// Self-profiling counters for overhead breakdown (cumulative, nanoseconds).
  struct overhead_counters {
    uint64_t enable_cb_ns;     ///< Time in cuptiEnableCallback calls (stage_begin/end)
    uint64_t callback_ns;      ///< Time in subscriber callback body
    uint64_t flush_ns;         ///< Time in cuptiActivityFlushAll
    uint64_t populate_ns;      ///< Time in populate_stats total
    uint32_t num_callbacks;    ///< Number of callback invocations
    uint32_t num_buf_req;      ///< Number of buffer_requested calls
    uint32_t num_buf_comp;     ///< Number of buffer_completed calls
    uint32_t num_kern_records; ///< Total kernel activity records delivered
  };

  [[nodiscard]] overhead_counters get_overhead_counters() const;
  void reset_overhead_counters();

 private:
  struct impl;
  std::unique_ptr<impl> _pimpl;
};

}  // namespace io
}  // namespace CUDF_EXPORT cudf
