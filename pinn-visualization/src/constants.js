// Placeholder option lists for the control bar.
// These mirror the model variants trained in /reference and saved in /weights —
// update the `value` fields to match whatever keys the backend API ends up expecting.

export const MODEL_OPTIONS = [
  { value: 'nn', label: 'Neural Network (NN)' },
  { value: 'vanilla-pinn', label: 'Vanilla PINN' },
  { value: 'adaptive-residual', label: 'PINN — Adaptive Residual Sampling' },
  { value: 'time-window', label: 'PINN — Time Window Marching' },
]

export const COLOR_SCHEME_OPTIONS = [
  { value: 'viridis', label: 'Viridis' },
  { value: 'plasma', label: 'Plasma' },
  { value: 'inferno', label: 'Inferno' },
  { value: 'magma', label: 'Magma' },
  { value: 'coolwarm', label: 'Coolwarm' },
  { value: 'rdbu', label: 'RdBu' },
]

// Initial condition is fixed for now rather than user-selectable.
export const INITIAL_CONDITION = 'planar-wave'

// Reported training stats per model (Table 7.1 — Overall performance
// comparison between models).
export const MODEL_STATS = {
  nn: { trainingPoints: 1000, accuracy: 0.93 },
  'vanilla-pinn': { trainingPoints: 200, accuracy: 0.93 },
  'time-window': { trainingPoints: 2000, accuracy: 0.8 },
  'adaptive-residual': { trainingPoints: 200, accuracy: 0.94 },
}
