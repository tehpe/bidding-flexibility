## Data

Raw input data is not stored in this repo.

raw_data:
  - file: RESULT_OVERVIEW_CAPACITY_MARKET_aFRR_2021.xlsx
    source: Regelleistung
    url: "https://www.regelleistung.net/apps/datacenter/tendering-files/?productTypes=aFRR&markets=CAPACITY&fileTypes=RESULTS&dateRange=2021-01,2021-12"
    accessed: "2025-03-17"
    used_for: marginal_prices

  - file: RESULT_LIST_ANONYM_ENERGY_MARKET_aFRR_2021.xlsx
    source: Regelleistung
    url: "https://www.regelleistung.net/apps/datacenter/tendering-files/?productTypes=aFRR&markets=ENERGY&fileTypes=ANONYMOUS_LIST_OF_BIDS&dateRange=2021-01,2021-12"
    accessed: "2025-03-17"
    used_for: rem_offers

  - file: SECONDS_BASE_AFRR_TARGET_VALUES_2021.csv
    source: Netztransparenz
    url: "https://www.netztransparenz.de/en/Balancing-Capacity/Balancing-Capacity-data/Data-in-second-resolution"
    accessed: "2025-03-17"
    used_for: activation_timeseries

These files should be placed in `data/tmp-raw/` before running:

```bash
python scripts/prepare_data.py