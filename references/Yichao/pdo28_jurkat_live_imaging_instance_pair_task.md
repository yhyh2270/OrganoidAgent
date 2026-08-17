# PDO28 and Jurkat Live-Imaging Instance-Pair Task

## Data
- Source LIF: `/home/lachlan/Downloads/PDO28 and Jurkat.lif`
- Repo copy: `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-9/PDO28 and Jurkat.lif`
- Dataset name: `Data-Yichao-9`
- Biological task: PDO28 organoid and S100A4 CAR-Jurkat live imaging.

## LIF structure
The LIF contains 6 series.

Static reference/control series:
- Series 0: `PDO28/Sigma Antibody 40X_1`, 2048x2048, z=1, t=1, channels=2, LUTs blue/green.
- Series 1: `PDO28/Sigma Ab 40X_1`, 2048x2048, z=1, t=1, channels=2, LUTs blue/green.
- Series 2: `PDO28/Sigma Ab 40X_2`, 2048x2048, z=1, t=1, channels=2, LUTs blue/green.

Live-imaging series to use for paired instance work:
- Series 3: `PDO28/S100A4_CAR-Jurkat D2`, 512x512, z=15, t=116, channels=2.
- Series 4: `PDO28/S100A4_CAR-Jurkar D3/Position001`, 512x512, z=13, t=116, channels=2.
- Series 5: `PDO28/S100A4_CAR-Jurkar D3/Position002`, 512x512, z=13, t=116, channels=2.

Live channel mapping:
- `c0`: green fluorescence, S100A4 CAR-Jurkat / T-cell signal.
- `c1`: gray transmitted/brightfield-like channel, used for organoid segmentation.

Time information:
- Live series have 116 time points.
- Metadata complete duration is approximately 72,215 seconds, about 20.06 hours.
- Cycle interval is approximately 627.9 seconds, about 10.47 minutes per frame.

## Current requested processing mode
Use the existing Yichao instance-pair pipeline, not a new one-off script.

The pipeline should:
- Export the LIF into JPEG planes.
- Organize JPEG planes by live series/position.
- Segment organoids from `c1`.
- Pair each segmented organoid crop with the corresponding green fluorescence `c0`.
- Save image-level intermediates, instance crops, overlays, and records.
- Maintain the same SQLite database format as the existing Yichao instance-pair datasets.
- Update the resized 256x256 paired dataset incrementally.
- Generate random pair previews for this dataset.

Important filter:
- Only series indices `3`, `4`, and `5` should be used for `Data-Yichao-9`.
- Static Sigma antibody reference series `0`, `1`, and `2` are not live c1/c0 organoid-pair data and should not enter the paired-instance database.

## Old-script command
Run:

```bash
bash analysis-tools/yichao_instance_pairs/run_yichao_dataset_incremental_pipeline.sh \
  --dataset-name Data-Yichao-9 \
  --lif-path "/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-9/PDO28 and Jurkat.lif"
```

Expected source folders after extraction:
- Flat JPEG export: `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-9/PDO28 and Jurkat_jpeg_all`
- Grouped live/position folders: `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-9/PDO28 and Jurkat_jpeg_all_by_position`

Expected analysis outputs:
- Instance-pair root: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs`
- SQLite database: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs/database/instance_pairs.sqlite`
- Resized 256 dataset: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs_resized_256`
- Random previews: under `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_instance_pairs_resized_256/previews`

## Live movie and T-cell mode-of-action analysis
The same exported JPEG planes can also support live movie generation:
- For each live series/position, create green max-intensity projection over z per time point from `c0`.
- Create gray/green overlay movies using `c1` as morphology/background and `c0` as T-cell fluorescence.
- Suggested output folder: `/home/lachlan/ProjectsLFS/OrganoidAgent/analysis-outputs/yichao_pdo28_jurkat_live_imaging`

Analysis goals for green T-cell mode of action:
- Track green T-cell fluorescence over time.
- Measure total green area and intensity near each organoid.
- Quantify cell/organoid contact, accumulation, infiltration, and possible killing-associated changes.
- Compare D2 vs D3 positions.
- Use the paired-instance database to associate each organoid crop with corresponding green fluorescence dynamics.

Do not infer biological mechanism from fluorescence alone. Treat the first pass as quantitative descriptive analysis: localization, contact, persistence, infiltration, and intensity dynamics.
