# ICE-Bench

This task folder provides a lightweight ICE-Bench integration path for smoke validation.

- Task: `ice_bench`
- Source: official ICE-Bench dataset payload format (`ali-vilab/ICE-Bench`)
- Dataset file expected by YAML: `/tmp/ice_bench_smoke.jsonl`

`examples/models/openrouter_ice_smoke.sh` can bootstrap one official sample into that file and run end-to-end image generation/editing smoke with local artifact saving.
