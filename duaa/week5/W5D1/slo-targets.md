# Service indicators and proposed targets

Team: <team name>
Use case: <what your service does and who uses it>
Service measured: <model, endpoint and namespace; state baseline if applicable>
Workload: <request shape, caller count, output limit or batch size>
Measurement period: <start and end, including date and timezone>
Instrumentation gaps: <what you cannot yet measure; write none if justified>

Complete two SLI sections. Copy one section if you choose a third. Use the exact
stat-panel title for Panel. Keep the field labels so the verifier can read them.
Support each target with observed measurements.

## SLI 1

Indicator: <what is measured, including percentile where relevant>
Panel: <exact title of its stat panel in Team service>
Unit: <seconds, requests/min, percent, or another explicit unit>
Target: <comparison and numeric value, or numeric operating range; label diagnostic thresholds>
Window: <query window and, if different, the SLO window>
Observed: <numeric result in the stated unit, with its workload and coverage>
Evidence: <query reference or saved result and measurement timestamp>
Why it fits: <why this measurement matters for this service and its users>
Limitations: <measurement boundary, sample size, and what needs retesting>

## SLI 2

Indicator: <what is measured, including percentile where relevant>
Panel: <exact title of its stat panel in Team service>
Unit: <seconds, requests/min, percent, or another explicit unit>
Target: <comparison and numeric value, or numeric operating range; label diagnostic thresholds>
Window: <query window and, if different, the SLO window>
Observed: <numeric result in the stated unit, with its workload and coverage>
Evidence: <query reference or saved result and measurement timestamp>
Why it fits: <why this measurement matters for this service and its users>
Limitations: <measurement boundary, sample size, and what needs retesting>
