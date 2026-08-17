# Yichao v2 Data Preview: N39Rep Goblet/Globet DF

Date prepared: 2026-06-03
Updated: 2026-06-27

## Source Files

- D2 source: `/home/lachlan/Downloads/N39Rep_Globet DF_D2.lif`
- D3 source: `/home/lachlan/Downloads/N39Rep_Globet_DF_D3.lif`
- D3_1 source: `/home/lachlan/Downloads/N39Rep_Globet DF_D3_1.lif`
- D3_2 source: `/home/lachlan/Downloads/N39Rep_Globet DF_D3_2.lif`
- D4 source: `/home/lachlan/Downloads/N39Rep_Globet DF_D4.lif`

User note for D3:

> 对我这次一共拍了6个视野，所以我要标注12345666组。然后每一组我都有用4个通道去拍他的名场。这四个通道我上面也有标注它的通道的名称，比如说那个。De那就是de这个通道，如果是Alex 488, 那就Alex 488这个通道一共有四个通道，就是我们的摄像机一共有四个通道。然后只有绿色的就是AX 488那个通道才有荧光，其他的通道是没有的。

Working interpretation:

- The four named acquisition channels are `ALEXA488`, `DAPI`, `ALEXA647`, and `mCherry`.
- Only `ALEXA488` should contain the biologically relevant green fluorescence.
- Each named acquisition series contains two internal extracted channels, `c0` and `c1`.
- From the preview, `ALEXA488 c0` is the green fluorescence-like image and `ALEXA488 c1` is the brightfield-like companion image.

This differs from older Yichao data where the usual direct pair was one folder/object with `c0 = fluorescence` and `c1 = brightfield`. In this v2 data, the same field is repeated under multiple named acquisitions, each with its own `c0/c1` pair.

## Local Data Layout

The raw/extracted data are local and git-ignored:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/DATA-Yichao-v2/
  1_N39Rep_Globet_DF_D3/
    N39Rep_Globet_DF_D3.lif
    N39Rep_Globet_DF_D3_jpeg_all/
    N39Rep_Globet_DF_D3_jpeg_all_by_position/
  2_N39Rep_Globet_DF_D4/
    N39Rep_Globet_DF_D4.lif
    N39Rep_Globet_DF_D4_jpeg_all/
    N39Rep_Globet_DF_D4_jpeg_all_by_position/
  3_N39Rep_Globet_DF_D2/
    N39Rep_Globet_DF_D2.lif
  4_N39Rep_Globet_DF_D3_1/
    N39Rep_Globet_DF_D3_1.lif
  5_N39Rep_Globet_DF_D3_2/
    N39Rep_Globet_DF_D3_2.lif
```

The D4 filename was copied with a sanitized space-free name inside the repo:

```text
N39Rep_Globet DF_D2.lif -> N39Rep_Globet_DF_D2.lif
N39Rep_Globet DF_D3_1.lif -> N39Rep_Globet_DF_D3_1.lif
N39Rep_Globet DF_D3_2.lif -> N39Rep_Globet_DF_D3_2.lif
N39Rep_Globet DF_D4.lif -> N39Rep_Globet_DF_D4.lif
```

## Extraction Summary

Extraction used:

```bash
/home/lachlan/miniconda3/envs/organoid/bin/python BioAgentUtils/lif_to_jpeg.py <input.lif> -o <stem>_jpeg_all --quality 95
/home/lachlan/miniconda3/envs/organoid/bin/python BioAgentUtils/organize_lif_jpegs.py <stem>_jpeg_all -o <stem>_jpeg_all_by_position --mode link
```

Results:

| Dataset | LIF image series | Grouped folders | JPEG planes | Timepoints | Internal channels per series |
|---|---:|---:|---:|---:|---:|
| D3 | 40 | 40 | 2760 | 1 | 2 |
| D4 | 24 | 24 | 952 | 1 | 2 |

The 2026-06-27 copied D2/D3_1/D3_2 files have not yet been extracted or segmented in this pass. Metadata inspection only:

| Dataset folder | Day hint | LIF image series | Field/series count | Timepoints | Z range | Internal channels per series |
|---|---|---:|---:|---:|---|---:|
| `3_N39Rep_Globet_DF_D2` | D2 | 15 | 15 generic `Series###` | 1 | 1 | 7 |
| `4_N39Rep_Globet_DF_D3_1` | D3 | 10 | 10 generic `Series###` | 1 | 4-13 | 8 |
| `5_N39Rep_Globet_DF_D3_2` | D3 | 12 | 12 generic `Series###` | 1 | 12-31 | 8 |

Metadata outputs:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_v2_metadata/yichao_v2_lif_metadata_summary.json`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_v2_metadata/yichao_v2_lif_series_metadata.csv`

Important: D2/D3_1/D3_2 use generic series names and 7/8 internal channels, unlike the already-previewed D3/D4 files. They require a channel-mapping preview before they are added to segmentation/training.

## Experiment Structure

### D3

ReadLIF metadata shows 40 named image series:

- 10 field IDs: `1` through `10`
- 4 named acquisitions per field: `ALEXA488`, `DAPI`, `ALEXA647`, `mCherry`
- Field 2 has one metadata typo: `ALEZA647`, normalized in the preview script to `ALEXA647`
- Each named acquisition has two extracted channels: `c0` and `c1`
- No time series: `t = 1`

Z-depths by field:

| Field | Z planes |
|---:|---:|
| 1 | 34 |
| 2 | 36 |
| 3 | 34 |
| 4 | 29 |
| 5 | 49 |
| 6 | 37 |
| 7 | 26 |
| 8 | 36 |
| 9 | 44 |
| 10 | 20 |

Important discrepancy to confirm: the user note says D3 has 6 fields of view, but the LIF metadata contains 10 numbered fields.

### D4

ReadLIF metadata shows 24 named image series:

- 6 field IDs: `1` through `6`
- 4 named acquisitions per field: `ALEXA488_BF`, `mCherry_BF`, `ALEXA647_BF`, `DAPI_BF`
- Each named acquisition has two extracted channels: `c0` and `c1`
- No time series: `t = 1`

Z-depths by field:

| Field | Z planes |
|---:|---:|
| 1 | 11 |
| 2 | 13 |
| 3 | 13 |
| 4 | 21 |
| 5 | 23 |
| 6 | 38 |

## Preview Outputs

Preview figures are in:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_v2_preview/
```

Files:

- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_v2_preview/yichao_v2_d3_all_fields_channel_preview.png`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_v2_preview/yichao_v2_d4_all_fields_channel_preview.png`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_v2_preview/yichao_v2_brightfield_candidate_comparison.png`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_v2_preview/yichao_v2_preview_summary.json`

Preview script:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-tools/yichao_v2/preview_extracted_channels.py
```

Preparation scripts:

```text
/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-tools/yichao_v2/inspect_v2_lif_metadata.py
/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-tools/yichao_v2/run_yichao_v2_lif_prepare.sh
```

## Provisional Channel Use For Later Modeling

Do not mix all four named acquisition folders as independent fluorescence targets without checking the biology. A safer first mapping is:

- Input brightfield: `ALEXA488 c1`
- Target fluorescence: `ALEXA488 c0`

The other named acquisitions can be used as controls or excluded:

- `DAPI c0`, `ALEXA647 c0`, and `mCherry c0` are expected to have no target fluorescence for this task.
- Their `c1` channels appear BF-like and may be redundant repeated brightfield captures of the same field.

Next decision after visual inspection:

- If `ALEXA488 c1` is visually the best BF channel, use only `ALEXA488 c1 -> ALEXA488 c0` for B2F.
- If all `c1` images are equivalent BF captures, use `ALEXA488 c1` only to avoid duplicate leakage.
- If D3 fields `7` through `10` are not intended samples, exclude them before adding v2 to the training database.
