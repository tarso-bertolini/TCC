// Edge inference firmware for the microgrid PPO policy.
//
// Reads the four observation channels (battery SOC, demand, PV, tariff) from
// the bench sensors and writes the discrete dispatch decision over GPIO/UART
// back to the battery controller. A 1 Hz loop matches the 1-hour MDP step
// downsampled for the test harness.

#include <Arduino.h>

#include "policy_inference.h"

namespace {
constexpr int kSampleHz = 1;
constexpr unsigned long kPeriodMs = 1000UL / kSampleHz;

PolicyInference g_policy;
unsigned long g_next_tick = 0;
unsigned long g_iteration = 0;

float readSocRatio()      { return analogRead(34) / 4095.0f; }
float readDemandKw()      { return analogRead(35) * (8.0f / 4095.0f); }
float readPvKw()          { return analogRead(32) * (5.0f / 4095.0f); }
float readTariffRate()    { return analogRead(33) * (0.50f / 4095.0f); }

void applyAction(int action) {
    // 0 = discharge, 1 = idle, 2 = charge
    digitalWrite(25, action == 2 ? HIGH : LOW);
    digitalWrite(26, action == 0 ? HIGH : LOW);
}
}  // namespace

void setup() {
    Serial.begin(115200);
    delay(150);
    Serial.println("[main] microgrid edge inference boot");

    pinMode(25, OUTPUT);
    pinMode(26, OUTPUT);
    analogReadResolution(12);

    if (!g_policy.begin()) {
        Serial.println("[main] policy init failed, halting");
        while (true) delay(1000);
    }
    g_next_tick = millis();
}

void loop() {
    if (millis() < g_next_tick) return;
    g_next_tick += kPeriodMs;

    const float obs[4] = {
        readSocRatio(),
        readDemandKw(),
        readPvKw(),
        readTariffRate(),
    };

    int action = g_policy.predict(obs);
    applyAction(action);

    Serial.printf(
        "step=%lu soc=%.3f demand=%.2f pv=%.2f tariff=%.3f -> action=%d (%lu us)\n",
        g_iteration++, obs[0], obs[1], obs[2], obs[3], action,
        static_cast<unsigned long>(g_policy.lastInferenceMicros()));
}
