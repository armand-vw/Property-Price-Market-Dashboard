# Datasets

## `ames.csv` — real housing data

A trimmed, schema-mapped extract of the **Ames Housing** dataset used for the
real-data benchmark and the notebooks.

- **Source:** Ames, Iowa residential property sales, originally compiled by
  Dean De Cock (2011) and distributed via
  [OpenML](https://www.openml.org/d/42165) as `house_prices`.
- **Records:** 1,460 real sales.
- **Columns** (mapped to the project schema): `market`, `neighborhood`,
  `bedrooms`, `bathrooms`, `sqft`, `lot_size`, `year_built`, `has_pool`,
  `garage_spaces`, `price`.
- **Mapping:** `bedrooms=BedroomAbvGr`, `bathrooms=FullBath+0.5·HalfBath`,
  `sqft=GrLivArea`, `lot_size=LotArea`, `year_built=YearBuilt`,
  `has_pool=PoolArea>0`, `garage_spaces=GarageCars`, `price=SalePrice`.
- **Regenerate:** `python scripts/fetch_real_dataset.py` (requires network).

This dataset is a public teaching dataset and is included here for
demonstration. It is separate from the synthetic, market-anchored listings that
power the shipped dashboard.
