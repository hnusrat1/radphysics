# Code audit of AI answers: method classification and physics checklist

You are auditing what an AI model did in a Python sandbox while answering a radiation-physics question. You read
its transcripts; you do not judge whether its numbers are right (that is scored elsewhere). Report what the code
does, with evidence. Do not modify anything under the repository; everything there is read-only data.

## The problem the model was given (configuration R0)

- Water cube 30 x 30 x 30 cm, density 1.000, vacuum outside.
- Point source on the central axis, 100 cm from the phantom surface (SSD 100 cm), photons emitted uniformly in solid
  angle inside the pyramid that makes a 10 x 10 cm field at the surface. Two lines, 1.1732 and 1.3325 MeV, equal
  probability. No collimator or other scatter.
- Point of interest: central axis, 5 cm depth. Scoring volume: cylinder on the axis, radius 1.0 cm, thickness 0.2 cm,
  centred at 5 cm depth.
- Phi0: primary photon fluence at the point (105 cm from the source) with the phantom removed. Most answers are per Phi0.
- Quantities asked (33): photon fluence (primary, scattered, spectra, angular), interaction fractions by process and
  by primary/scattered parent, energy-transfer fractions, initial electron spectrum and direction, electron fluence
  spectrum and angular distribution, delta-ray fraction, radiative-to-collision ratio, D/Phi0, K/Phi0, Kcol/Phi0,
  beta, g, s_w,air, PDD(5).
- Conditions: A1 = Python with numpy/scipy, no data files, no internet, no transport codes. A2 = A1 plus two data
  files in /tmp/data: xcom_water.txt (photon cross sections) and estar_water.txt (electron stopping powers).
- Sandbox limits: 4 cores, 10 minutes per call, at most 30 tool calls per conversation.

## Where the evidence is

Each conversation is a directory `ai/raw/M4-R0/<slot>/R0_<A1|A2>_observables_r<k>/` (paths relative to
`/home/user/radphysics/co60-refpoint`) with:
- `tool_calls/NNN.py`: code the model ran, in order; `tool_calls/NNN.out.json`: stdout, stderr, exit code, wall time.
  Some models write a script file from one call (e.g. `open('/tmp/work/sim.py','w').write(...)`) and run it with
  subprocess in a later call; read the written script text inside the earlier call.
- `final.txt`: the model's final reply; `parsed.json`: the validated answer, including a `method` field (the model's
  own description, which can be wrong: check it against the code).
- `meta.yaml`: outcome, tool-call count.

## For each conversation, fill one JSON object

```json
{
  "conversation": "ai/raw/M4-R0/<slot>/R0_A1_observables_r1",
  "method_class": "monte_carlo | hybrid | analytic | no_code",
  "method_summary": "one or two sentences: what the code computes and how the final numbers were produced",
  "answer_from_code": "yes | partly | no",
  "histories": 1234567,
  "geometry": "finite_30cm_cube | semi_infinite | infinite | slab | not_modelled | other",
  "source_model": "diverging_pyramid_ssd100 | parallel_beam | point_isotropic | not_modelled | other",
  "spectrum": "two_lines | mean_energy_1.25 | other",
  "compton": "klein_nishina_sampled | klein_nishina_analytic | other | not_modelled",
  "rayleigh": "sampled_with_angle | forward_no_deflection | ignored | absorbed_or_killed | analytic_only",
  "photoelectric": "absorption_modelled | ignored | analytic_only",
  "pair": "modelled_with_annihilation | modelled_no_annihilation | ignored | analytic_only",
  "binding_or_doppler": "yes | no",
  "photon_cutoff_keV": 10,
  "cross_section_source": "xcom_file | hardcoded_table | analytic_formula | mixed | none",
  "electron_treatment": "none | csda_slowing_down_spectrum | local_deposition | condensed_history_transport | other",
  "electron_direction": "sampled_from_kinematics | assumed_shape | transported | not_computed | other",
  "delta_rays": "computed | assumed_value | not_computed",
  "scoring": "track_length_in_cylinder | point_or_next_event_estimator | collision_estimator | analytic | other",
  "phi0_normalization": "consistent | inconsistent | unclear",
  "dose_method": "kcol_times_beta_cpe | energy_deposition_tally | kerma_only | assumed_or_recalled | other",
  "pdd_method": "simulated_depth_curve | recalled_or_assumed | analytic | other",
  "notable_physics_choices": "1 to 3 sentences on choices that matter for the physics (approximations, omissions, bugs)",
  "evidence": [
    {"file": "tool_calls/007.py", "line": 42, "quote": "exact short quote from the code or output"}
  ]
}
```

Definitions:
- `monte_carlo`: the photon quantities in the answer come from a stochastic photon-transport simulation the model ran
  (random free paths and interaction sampling, many histories). `hybrid`: a simulation was run but some photon
  quantities (say which in method_summary) come from analytic estimates or assumed values. `analytic`: deterministic
  formulas only (exponential attenuation, buildup factors, single-scatter integrals, assumed or recalled values),
  even if the code uses random numbers for something minor. `no_code`: no tool call contributed.
- `answer_from_code`: whether the submitted numbers are printed by the model's code (compare final answer values with
  the last relevant stdout), or partly typed in from elsewhere.
- `histories`: source photons actually simulated in the run(s) whose output fed the answer; null if not a simulation.
- `photon_cutoff_keV`: energy below which photons are killed or deposited locally; null if none or not a simulation.
- `phi0_normalization`: whether the fluence tallies and Phi0 are on the same per-source-photon basis. A useful check:
  for this geometry the primary transmission at 5 cm, fluence/Phi0 for primary photons, is about 0.73; a value far
  from that points to an inconsistent normalization.
- `electron_direction`: how the initial electron direction (Layer 3) and the electron fluence angular distribution
  (Layer 4) were produced.
- Every claim about method_class, photon_cutoff_keV, electron_treatment, scoring and phi0_normalization must have at
  least one evidence entry with an exact quote. If you cannot find evidence, use "other"/null and say so in
  notable_physics_choices. Do not guess.

## Output

Write a JSON array with one object per conversation, in conversation order, to the output path you are given, with
`json.dump(..., indent=1)`. Validate that it loads. Then reply with a short summary: the method class of each
conversation, and anything surprising (bugs, normalization errors, answers not produced by the code).
