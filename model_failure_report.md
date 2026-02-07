# Model Failure Report — Board Game Trend Analysis

The model reported 0.0312 for the mean absolute change in average rating between the earliest and latest snapshots, but the correct value is 0.0416, likely because the model misidentified `bgg_db_2018_01.xlsx` (January 2018) as the latest snapshot instead of `bgg_db_1806.xlsx` (June 2018), thereby computing the change only across the 2017-04 → 2018-01 window rather than the full 2017-04 → 2018-06 span as required by the task.

The model reported a Pearson correlation coefficient of 0.5625 between average weight and average rating in the most recent snapshot, but the correct value is 0.5680, likely because the model computed the correlation on the `bgg_db_2018_01.xlsx` data (January 2018) instead of the true most-recent snapshot `bgg_db_1806.xlsx` (June 2018), which has a different weight-rating relationship due to updated game entries and rating shifts over the intervening five months.

The model reported game_id 186751 (Mythic Battles: Pantheon) as the highest-ranked game in the most recent snapshot, but the correct value is game_id 174430 (Gloomhaven), likely because the model ranked games within the `bgg_db_2018_01.xlsx` snapshot (January 2018) rather than the actual latest snapshot `bgg_db_1806.xlsx` (June 2018), where Gloomhaven's average rating of 8.98893 places it unambiguously at rank 1.

The model reported 1349.0066 for the population standard deviation of playtime among the top-20 games, but the correct value is 1257.7396, likely because the model derived its top-20 set from the wrong snapshot (`bgg_db_2018_01.xlsx` instead of `bgg_db_1806.xlsx`), resulting in a different set of 20 games with different avg_time values and therefore a different population standard deviation.

The model reported 6.7894 for the skewness of the num_votes distribution, but the correct value is 6.8366, likely because the model computed skewness using only a single snapshot (`bgg_db_2018_01.xlsx`) rather than across all retained games in all three snapshots combined, as specified by the task's instruction to summarize community engagement "across all retained games."
