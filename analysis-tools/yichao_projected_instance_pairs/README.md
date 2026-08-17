# Yichao Projected-Z Instance Pairs

This folder is a separate projected-z pipeline. It does not write into or modify:

- `analysis-tools/yichao_instance_pairs/`
- `analysis-outputs/yichao_instance_pairs/`

Default output root:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_projected_instance_pairs`

## What It Does

For every paired Yichao brightfield/fluorescence stack grouped by:

```text
dataset / object-or-position / time_index
```

the pipeline:

1. collects all z-planes for `c1` brightfield and matched `c0` fluorescence,
2. creates a projected brightfield image and projected fluorescence image,
3. segments the projected brightfield image,
4. saves full-image intermediates and instance crops,
5. writes a new projected-z SQLite database.

Default projection:

- brightfield: `min` projection, to preserve dark organoid structures across z,
- fluorescence: `max` projection, to preserve positive fluorescence signal across z.

## Main Output

```text
analysis-outputs/yichao_projected_instance_pairs/
  images/
  instances/
  projections/
  failures/
  manifests/
  database/projected_instance_pairs.sqlite
```

## Run

In tmux:

```bash
bash analysis-tools/yichao_projected_instance_pairs/resume_yichao_projected_pairs_tmux.sh
```

Directly:

```bash
bash analysis-tools/yichao_projected_instance_pairs/run_yichao_projected_instance_pair_pipeline.sh
```

## Database Tables

- `projected_images`: one row per projected `(dataset, object_name, time_index)`.
- `projected_instances`: one row per segmented projected organoid instance.
- `metadata`: summary JSON.

The projected instance table includes:

- brightfield crop path,
- fluorescence crop path,
- mask crop path,
- `is_edge_padded`,
- dataset,
- object name,
- experiment label,
- experiment design,
- replicate label,
- sample label,
- day label/index,
- position label/index,
- time index,
- z-index list and z-count.

