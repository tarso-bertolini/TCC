// Edge inference wrapper around the quantised PPO policy.
//
// Exposes a single PolicyInference class. The microgrid telemetry observation
// vector is fed in as four floats (soc_ratio, demand_kw, pv_kw, tariff_rate)
// and the discrete dispatch action {0=discharge, 1=idle, 2=charge} is
// returned.

#pragma once

#include <cstdint>

class PolicyInference {
public:
    PolicyInference();
    bool begin();
    int predict(const float observation[4]);
    uint32_t lastInferenceMicros() const { return last_inference_us_; }

private:
    static constexpr int kArenaBytes = 32 * 1024;
    alignas(16) uint8_t tensor_arena_[kArenaBytes];
    uint32_t last_inference_us_;
    bool ready_;
};
