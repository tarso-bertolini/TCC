#include "policy_inference.h"
#include "ppo_model_data.h"

#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

namespace {
constexpr int kObsLen = 4;
constexpr int kNumActions = 3;

const tflite::Model* g_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;
tflite::AllOpsResolver g_resolver;
}  // namespace

PolicyInference::PolicyInference() : last_inference_us_(0), ready_(false) {}

bool PolicyInference::begin() {
    g_model = tflite::GetModel(g_ppo_model_data);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("[policy] tflite schema mismatch");
        return false;
    }

    static tflite::MicroInterpreter static_interpreter(
        g_model, g_resolver, tensor_arena_, kArenaBytes);
    g_interpreter = &static_interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("[policy] AllocateTensors failed");
        return false;
    }

    g_input = g_interpreter->input(0);
    g_output = g_interpreter->output(0);
    ready_ = true;
    Serial.printf("[policy] ready (arena=%d B, model=%d B)\n", kArenaBytes, g_ppo_model_data_len);
    return true;
}

int PolicyInference::predict(const float observation[4]) {
    if (!ready_) return 1;  // safe default = idle

    // Quantise from float -> int8 using the input tensor's scale + zero point.
    const float scale = g_input->params.scale;
    const int zero_point = g_input->params.zero_point;
    for (int i = 0; i < kObsLen; ++i) {
        int q = static_cast<int>(roundf(observation[i] / scale)) + zero_point;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        g_input->data.int8[i] = static_cast<int8_t>(q);
    }

    const uint32_t t0 = micros();
    TfLiteStatus status = g_interpreter->Invoke();
    last_inference_us_ = micros() - t0;
    if (status != kTfLiteOk) {
        Serial.println("[policy] Invoke failed");
        return 1;
    }

    int best = 0;
    int8_t best_val = g_output->data.int8[0];
    for (int a = 1; a < kNumActions; ++a) {
        if (g_output->data.int8[a] > best_val) {
            best_val = g_output->data.int8[a];
            best = a;
        }
    }
    return best;
}
