# Expected Cost And Time

These are rough planning estimates. Actual time and cost depend on GPU model, provider image, CPU/data-loading speed, storage performance, and whether the instance is shared or throttled.

- This batch contains only two Pinyin-Diacritic matched-data runs:
  - `diacritic_125m_b1024_matched_data_4epoch_seed43`
  - `diacritic_125m_b1024_matched_data_4epoch_seed44`
- Each run uses `max_steps=29764`, `tokens_per_update=65536`, and expected `tokens_seen=1950613504`.
- The previously observed matched-data seed42 run took about 6.2 hours on the A100-class instance.
- Two matched-data extra seed runs should therefore take about 12.4 to 12.6 hours of training time on a similar A100-class instance, plus setup, smoke tests, archiving, and transfer.
- At `$0.529/hour`, expected compute cost is approximately `$6.6 to $6.8`, excluding data transfer, setup time, idle time, storage, and provider overhead.

The target new instance has a 30GB disk. The workflow still checks free disk before each run and uses a default `MIN_FREE_DISK_GB=10`. The prior matched-data/tar workflow used roughly 4GB per completed run including the unpacked output and tarball, so two runs should fit, but the script will still stop if the free-space check fails.
