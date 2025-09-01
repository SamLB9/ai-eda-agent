# TODO List

## Completed Tasks
- [x] Fix descriptive statistics table showing same value (232725.00) in all columns
- [x] Move correlation matrix from Dataset Summary to Statistical Insights section
- [x] Add feature names as first column in correlation matrix for better readability
- [x] Fix correlation table titles and prevent duplicates
- [x] Fix repeated sentences in Final Explanation section
- [x] Fix 'Column not found: genre' error in multiprocessing execution
- [x] Remove absolute correlation tables and keep only simple correlation with popularity
- [x] Fix repeated sentences in Final Explanation section (e.g., Recommended next steps)
- [x] Fix IndentationError in session_runner.py line 1216
- [x] Make data tables fit content height instead of having fixed height with empty rows
- [x] Fix DataFrame data type conversion and increase execution timeout to prevent 30s timeout
- [x] Fix ValueError: Length mismatch in groupby column assignment when agg() creates MultiIndex columns
- [x] Optimize LLM-generated code performance by adding sampling requirements and increasing timeouts
- [x] Fix ValueError in seaborn boxplot when sampling causes insufficient data per category
- [x] Fix sanitizer that's being too aggressive and breaking LLM-generated code
- [x] Fix syntax errors in session_runner.py caused by orphaned code from aggressive sanitizer
- [x] Fix IndentationError in boxplot code caused by aggressive sanitizer regex replacements
- [x] Replace 'object' with 'text' for string columns in Column Data Types table to make it more user-friendly
- [x] Fix EDA hanging at data type conversion step by removing slow pd.to_numeric() operations
- [x] Fix EDA hanging at exec() call by adding auto-sampling for large datasets and debug output
- [x] Add detailed debug messages outside redirect_stdout to track where EDA execution hangs
- [x] Fix LLM prompt to ensure consistent use of sampled dataset throughout all operations to prevent 600s timeouts
- [x] Revert sampling changes to use full dataset for complete EDA analysis while optimizing performance
- [x] Update LLM prompt to ensure all plots have descriptive titles and axis labels for better readability
- [x] Add pre-loaded datasets (Spotify, Titanic, Video Game Sales) to Streamlit app with dataset selection dropdown
- [x] Fix dataset loading paths using robust pathlib-based resolution
- [x] Prepare beta version by commenting out advanced time budget options and renaming 5min option

## Pending Tasks
- [ ] Fix EDA hanging at exec() call - investigate multiprocessing context, import restrictions, and environment differences
- [ ] Add more comprehensive error handling for dataset loading failures
- [ ] Consider adding dataset preview functionality in the sidebar
- [ ] Add dataset metadata display (column descriptions, data types, etc.)
- [ ] Re-enable advanced time budget options (10, 20, 30 min) for full release
- [ ] Add ML model training capabilities for longer time budgets
