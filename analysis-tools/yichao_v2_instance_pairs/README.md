# Yichao v2 Instance-Pair Pipeline

This pipeline prepares per-organoid instance crops for `DATA-Yichao-v2` using the same segmentation logic as the v1 Yichao instance-pair pipeline, with an expanded channel model.

## Main Supervised Pair

For named acquisition folders:

```text
input  = ALEXA488 c1 brightfield/reference
target = ALEXA488 c0 green fluorescence
```

This is based on Yichao's note that the microscope used four named channels and only `ALEXA488` has the green fluorescence signal.

## Auxiliary Channels Preserved

The pipeline also saves other channel crops for later modeling:

```text
DAPI c0
ALEXA647 c0
mCherry c0
```

and their brightfield/reference companions when available. These are stored in JSON columns and as crop files so later models can test:

```text
brightfield + auxiliary autofluorescence / marker channels -> ALEXA488 green
```

## Generic Numeric Channel Maps

For the generic `Series###` LIFs, the mapping is inferred from numeric-channel previews and recorded with confidence:

| Dataset folder | Main BF | Main green target | Confidence |
| --- | --- | --- | --- |
| `3_N39Rep_Globet_DF_D2` | `c4` | `c3` | `inferred_medium` |
| `4_N39Rep_Globet_DF_D3_1` | `c3` | `c2` | `inferred_high` |
| `5_N39Rep_Globet_DF_D3_2` | `c3` | `c2` | `inferred_high` |

The database includes `channel_map_confidence` so downstream training can filter to high-confidence rows only.

## Run

```bash
cd /home/lachlan/ProjectsLFS/OrganoidAgent
analysis-tools/yichao_v2_instance_pairs/resume_yichao_v2_instance_pairs_tmux.sh
```

Main output:

```text
analysis-outputs/yichao_v2_instance_pairs/database/yichao_v2_instance_pairs.sqlite
analysis-outputs/yichao_v2_instance_pairs/manifests/image_records.csv
analysis-outputs/yichao_v2_instance_pairs/manifests/instance_records.csv
analysis-outputs/yichao_v2_instance_pairs/manifests/summary.json
```
