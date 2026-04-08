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
 * External correlation IDs are used to map kernel records to pipeline stages.
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
    bool use_hes{false};  ///< Enable HES (Blackwell+, CUDA 12.8+). Must be
                          ///< created before CUDA context if true.
  };

  explicit cupti_timing_observer(config cfg = {.use_hes = false});
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

 private:
  struct impl;
  std::unique_ptr<impl> _pimpl;
};

}  // namespace io
}  // namespace CUDF_EXPORT cudf
