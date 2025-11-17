# NBA Analysis Data Requirements

## Required Files

The analysis script expects three CSV files with specific structures:

### 1. Regular_Season.csv (Game-level data)
**Required columns (exact spelling):**
- `Season` - Season identifier (e.g., "2012-13", "2013-14")
- `Date` - Game date
- `HomeTeam` - Home team abbreviation (e.g., "LAL", "GSW")
- `AwayTeam` - Away team abbreviation
- `HomePTS` - Home team points scored
- `AwayPTS` - Away team points scored

**Example:**
```
Season,Date,HomeTeam,AwayTeam,HomePTS,AwayPTS
2012-13,2012-10-30,MIA,BOS,120,107
2012-13,2012-10-30,LAL,DAL,99,91
```

### 2. Playoffs.csv (Game-level data)
**Same structure as Regular_Season.csv:**
- `Season`
- `Date`
- `HomeTeam`
- `AwayTeam`
- `HomePTS`
- `AwayPTS`

### 3. nba.csv (Player-level data)
**Required columns:**
- `Player` - Player name
- `Team` - Team abbreviation
- `Season` (or `year`) - Season identifier
- At least one of:
  - `Minutes` (or `MIN`) - Total minutes played
  - `Games` (or `GP`) - Games played

**Example:**
```
Player,Team,Season,Minutes,Games
LeBron James,MIA,2012-13,2877,76
Kevin Durant,OKC,2012-13,3119,81
```

## Current Data Issue

The current files in this directory (Regular_Season.csv, Playoffs.csv) contain **player statistics**, not **game-level data**. Please replace these files with game logs that match the format above.

## Analysis Features

Once the correct data is provided, the script will:
1. Calculate Simple Rating System (SRS) ratings for teams
2. Measure correlation between team ratings and win percentage
3. Analyze playoff performance shifts
4. Calculate home court advantage
5. Measure roster continuity
6. Create head-to-head playoff margin heatmap
7. Test if regular season strength predicts playoff success

## Running the Analysis

```bash
python nba_analysis.py
```

The script will generate:
- `analysis_summary.json` - All numerical results
- `playoff_margin_heatmap.png` - Visual heatmap
