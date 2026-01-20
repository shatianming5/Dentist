# Aggregate runs (prep2target)

- generated_at: 2026-01-17T15:26:46Z
- root: `/home/ubuntu/tiasha/dentist/runs/prep2target/v1`
- runs: 12
- seed_filter: [1337, 2020, 2021]

- groups: 4

| test_total (mean±std) | test_chamfer (mean±std) | test_margin (mean±std) | test_occlusion (mean±std) | n | dataset | exp | model | n_points | latent_dim | cond_label | lambda_margin | lambda_occlusion | clearance |
|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|---:|---:|---:|
| 0.0628±0.0004 | 0.0627±0.0004 | 0.0270±0.0006 | 0.0002±0.0000 | 3 | v1 | constraints_occlusion | p2t | 512 | 256 | plain | 0.0 | 0.1 | 0.5 |
| 0.0629±0.0003 | 0.0629±0.0003 | 0.0273±0.0001 | 0.0004±0.0001 | 3 | v1 | baseline | p2t | 512 | 256 | plain | 0.0 | 0.0 | 0.5 |
| 0.0659±0.0003 | 0.0633±0.0004 | 0.0258±0.0005 | 0.0004±0.0001 | 3 | v1 | constraints_margin | p2t | 512 | 256 | plain | 0.1 | 0.0 | 0.5 |
| 0.0661±0.0009 | 0.0631±0.0010 | 0.0262±0.0007 | 0.0003±0.0000 | 3 | v1 | multitask_constraints | p2t | 512 | 256 | plain | 0.1 | 0.1 | 0.5 |
