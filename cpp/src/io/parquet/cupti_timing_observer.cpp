/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/io/cupti_timing_observer.hpp>
#include <cudf/utilities/error.hpp>

#include <cupti.h>

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudf::io {

namespace {

// Check CUPTI result and throw on error, except for expected failures
#define CUPTI_CHECK(call)                                                              \
  do {                                                                                 \
    CUptiResult _status = (call);                                                      \
    CUDF_EXPECTS(_status == CUPTI_SUCCESS, "CUPTI error: " + std::to_string(_status)); \
  } while (0)

constexpr size_t kBufferSize = 8 * 1024 * 1024;  // 8 MB activity buffer

}  // namespace

struct cupti_timing_observer::impl {
  bool active = false;
  CUpti_SubscriberHandle subscriber{};

  // --- Correlation tracking via callback API ---
  // stage_begin/stage_end set the "current stage". The subscriber callback fires
  // synchronously on the same thread for every CUDA runtime API call, recording
  // the CUPTI-assigned correlation ID → our stage encoding.
  bool stage_active = false;
  uint64_t current_ext_id{};  // (stage << 32) | sub_stage_id

  // correlation_id → ext_id — built from callback, only accessed from user thread
  std::unordered_map<uint32_t, uint64_t> corr_to_ext;

  // --- Kernel records from Activity API ---
  // Written from CUPTI worker thread (buffer_completed), read from populate_stats.
  std::mutex mutex;

  struct kernel_record {
    uint32_t correlation_id;
    uint64_t start_ns;
    uint64_t end_ns;
  };

  std::vector<kernel_record> kernel_records;

  // Static instance pointer for activity buffer callbacks
  static impl* s_instance;

  static void CUPTIAPI subscriber_callback(void* userdata,
                                           CUpti_CallbackDomain domain,
                                           CUpti_CallbackId cbid,
                                           void const* cbdata)
  {
    auto* self = static_cast<impl*>(userdata);
    if (!self->stage_active) return;
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API) return;

    auto const* info = static_cast<CUpti_CallbackData const*>(cbdata);
    if (info->callbackSite != CUPTI_API_ENTER) return;

    // Map this CUDA API call's correlation ID to our current stage
    self->corr_to_ext[info->correlationId] = self->current_ext_id;
  }

  static void CUPTIAPI buffer_requested(uint8_t** buffer,
                                        size_t* size,
                                        size_t* max_num_records)
  {
    *buffer          = static_cast<uint8_t*>(malloc(kBufferSize));
    *size            = *buffer ? kBufferSize : 0;
    *max_num_records = 0;
  }

  static void CUPTIAPI buffer_completed(CUcontext ctx,
                                        uint32_t stream_id,
                                        uint8_t* buffer,
                                        size_t size,
                                        size_t valid_size)
  {
    if (!s_instance || !buffer) return;

    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS) {
      if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
        auto* kernel = reinterpret_cast<CUpti_ActivityKernel9*>(record);
        std::lock_guard lock(s_instance->mutex);
        s_instance->kernel_records.push_back(
          {kernel->correlationId, kernel->start, kernel->end});
      }
    }

    free(buffer);
  }
};

cupti_timing_observer::impl* cupti_timing_observer::impl::s_instance = nullptr;

cupti_timing_observer::cupti_timing_observer(config cfg) : _pimpl{std::make_unique<impl>()}
{
  // Set static instance for buffer callbacks
  impl::s_instance = _pimpl.get();

  // Try to subscribe. If another subscriber exists, degrade gracefully.
  CUptiResult res = cuptiSubscribe(
    &_pimpl->subscriber, impl::subscriber_callback, _pimpl.get());
  if (res == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) {
    _pimpl->active = false;
    return;
  }
  CUPTI_CHECK(res);

  // Enable runtime API callback domain so subscriber_callback fires
  // for CUDA API calls, letting us map correlation IDs to stages.
  CUPTI_CHECK(cuptiEnableDomain(1, _pimpl->subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));

  // Register activity buffer callbacks and enable kernel activity
  CUPTI_CHECK(cuptiActivityRegisterCallbacks(impl::buffer_requested, impl::buffer_completed));
  CUPTI_CHECK(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

  // Optionally enable HES for Blackwell+ (CUDA 12.8+).
  // Must be called before CUDA context creation for correct operation.
  // Falls back to legacy SW tracing on failure.
  if (cfg.use_hes) {
    CUptiResult hes_res = cuptiActivityEnableHWTrace(1);
    if (hes_res != CUPTI_SUCCESS) {
      // HES not available; legacy SW tracing remains active
    }
  }

  _pimpl->active = true;
}

cupti_timing_observer::~cupti_timing_observer()
{
  if (_pimpl->active) {
    cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiUnsubscribe(_pimpl->subscriber);
  }
  if (impl::s_instance == _pimpl.get()) { impl::s_instance = nullptr; }
}

void cupti_timing_observer::stage_begin(parquet_pipeline_stage stage,
                                        uint32_t sub_stage_id,
                                        rmm::cuda_stream_view)
{
  if (!_pimpl->active) return;
  _pimpl->current_ext_id = (static_cast<uint64_t>(stage) << 32) | sub_stage_id;
  _pimpl->stage_active   = true;
}

void cupti_timing_observer::stage_end(parquet_pipeline_stage stage,
                                      uint32_t sub_stage_id,
                                      rmm::cuda_stream_view)
{
  if (!_pimpl->active) return;
  _pimpl->stage_active = false;
}

void cupti_timing_observer::populate_stats(parquet_pipeline_stats& stats)
{
  if (!_pimpl->active) return;

  // Force-flush CUPTI activity buffers so all kernel records are delivered.
  // GPU work is already synchronized by the reader.
  cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);

  std::lock_guard lock(_pimpl->mutex);

  // Sum kernel durations per ext_id (encodes stage + sub_stage)
  std::map<uint64_t, uint64_t> duration_by_ext;
  for (auto const& kr : _pimpl->kernel_records) {
    auto it = _pimpl->corr_to_ext.find(kr.correlation_id);
    if (it != _pimpl->corr_to_ext.end()) {
      duration_by_ext[it->second] += (kr.end_ns - kr.start_ns);
    }
  }

  // Drain consumed records
  _pimpl->kernel_records.clear();
  _pimpl->corr_to_ext.clear();

  // Emit parquet_stage_timing entries
  for (auto const& [ext_id, total_ns] : duration_by_ext) {
    auto stage     = static_cast<parquet_pipeline_stage>(ext_id >> 32);
    auto sub_stage = static_cast<uint32_t>(ext_id & 0xFFFFFFFF);
    stats.stages.emplace_back(parquet_stage_timing{stage, sub_stage, total_ns});
  }
}

bool cupti_timing_observer::is_active() const { return _pimpl->active; }

}  // namespace cudf::io
