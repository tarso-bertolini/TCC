# Awesome Apple-Inspired UI Patterns

Designing a microgrid application with the elegance of Apple's ecosystem requires strict adherence to their principles.

## 1. Typography
*   **Font Family:** `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`
*   **Weights:** Light (300) for large headers, Regular (400) for body, Semibold (600) for emphasis/labels.
*   **Line-Height:** 1.5 for readability.

## 2. Layout & Spacing
*   **Grid System:** 12-column flexible grid.
*   **Corner Radii:** `16px` to `24px` for cards, `12px` for buttons. (Squircle shape).
*   **Paddings:** Generous paddings (e.g., `24px` or `32px` inside widget cards).

## 3. Colors & Materials
*   **Backgrounds:** Clean White (`#F5F5F7` off-white) or True Black for Dark Mode.
*   **Materials (Glassmorphism):** Use `backdrop-filter: blur(20px)` with translucent overlays (`rgba(255, 255, 255, 0.7)`) to create depth and hierarchy without heavy drop shadows.
*   **Accents:** System Blue (`#007AFF`) or vibrant gradients for key actions. Success Green for positive energy metrics (`#34C759`).

## 4. Components
*   **Dashboard Widgets:** Segmented controls to toggle modes (Comfort / Economy / Performance). Information should be glanceable.
*   **Graphs:** Smooth, curved line charts devoid of gridlines. Emphasize the data trend over precise axis ticks.
*   **Vortex Voice:** A floating, glowing orb indicator in the bottom corner when voice control ("Jarvis, ...") is actively listening or processing.

## 5. Views
1.  **Main Dashboard:** The live microgrid environment (Predictions, Live Tariffs, Voice Control).
2.  **Sandbox:** The experimental space routing to the Stable Baselines3 Python outputs via IPC (Inter-Process Communication).