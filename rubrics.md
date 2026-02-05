# NIRF Analysis Output Rubrics

---

## Section 1: Value Oriented Rubrics

**1. Reports the TLR coefficient from the 75th-percentile quantile regression of Score on predictors TLR, RPC, GO, and OI using NIRF 2025 data as 0.3637 rounded to four decimal places [0.3601, 0.3673].**
*35 points · Must Have*
**Proof of Ask:** *"Fit a 75th-percentile quantile regression with Score as the target and TLR, RPC, GO, OI as predictors using NIRF 2025 data. Extract the coefficient for TLR from the fitted parameters. Report quantile_regression_tlr_coefficient rounded to four decimal places."*

**2. Reports the validation R-squared for PLSRegression with n_components equal to 3 and scale set to False predicting Score from predictors TLR, RPC, GO, OI, and PERCEPTION on the concatenated NIRF 2023 and 2024 dataset with a 70/30 train-validation split as 0.9981 rounded to four decimal places [0.9881, 1.0000].**
*30 points · Must Have*
**Proof of Ask:** *"Fit PLSRegression with n_components equal to 3 and scale set to False on StandardScaler-transformed training data using predictors TLR, RPC, GO, OI, PERCEPTION to predict Score on combined 2023 and 2024 NIRF data. Calculate R-squared on the validation set. Report pls_validation_r_squared rounded to four decimal places."*

**3. Reports the first canonical correlation computed as Pearson correlation between the first transformed X component and the first transformed Y component from CCA with n_components equal to 2 where X contains TLR and GO and Y contains RPC and PERCEPTION using NIRF 2024 data as 0.4691 rounded to four decimal places [0.4591, 0.4791].**
*25 points · Must Have*
**Proof of Ask:** *"Fit CCA with n_components equal to 2 on independently standardised X containing TLR and GO and Y containing RPC and PERCEPTION using NIRF 2024 data. Compute Pearson correlation between the first transformed X and first transformed Y components. Report first_canonical_correlation rounded to four decimal places."*

**4. Reports the NMF relative reconstruction error computed as the Frobenius norm of the difference between the MinMax-scaled input matrix and the reconstructed matrix divided by the Frobenius norm of the MinMax-scaled input matrix for features TLR, RPC, GO, OI, and PERCEPTION from NIRF 2025 using NMF with n_components equal to 2 as 0.2669 rounded to four decimal places [0.2536, 0.2802].**
*25 points · Must Have*
**Proof of Ask:** *"Apply NMF with n_components equal to 2 and init set to nndsvd on MinMax-scaled NIRF 2025 data using features TLR, RPC, GO, OI, PERCEPTION. Compute relative reconstruction error as Frobenius norm of difference divided by Frobenius norm of original scaled matrix. Report nmf_relative_reconstruction_error rounded to four decimal places."*

**5. Reports the Wasserstein distance between the Score distributions from NIRF 2023 and NIRF 2025 as 2.6959 rounded to four decimal places [2.5611, 2.8307].**
*20 points · Must Have*
**Proof of Ask:** *"Apply scipy.stats.wasserstein_distance to quantify the minimum transport cost of transforming the 2023 Score distribution into the 2025 Score distribution. Report wasserstein_distance_2023_2025 rounded to four decimal places."*

**6. Reports the bootstrap confidence interval lower bound defined as the 2.5th percentile of 10000 bootstrap sample means of the per-institution Score difference computed as Score in 2025 minus Score in 2023 for institutions whose Institute ID is present in all three NIRF years as 1.7029 rounded to four decimal places [1.6178, 1.7880].**
*20 points · Must Have*
**Proof of Ask:** *"Match institutions by Institute ID across NIRF 2023, 2024, and 2025. Compute Score 2025 minus Score 2023 for each matched institution. Generate 10000 bootstrap sample means and extract the 2.5th percentile. Report bootstrap_ci_lower_bound rounded to four decimal places."*

---

## Section 2: Visualization Rubrics

---

### quantile_regression_residuals.png

**1. Includes a connected scatter plot for quantile regression diagnostic analysis for Score using predictors TLR, RPC, GO, and OI from NIRF 2025 with fitted values on the x-axis and residuals on the y-axis that is semantically similar to the attached visualization.**
*Must Have*
**Proof of Ask:** *"Generate a connected scatter plot for quantile regression diagnostic analysis with x-axis representing fitted values and y-axis representing residuals."*

**2. Provides a connected scatter plot for quantile regression diagnostic analysis for Score with fitted values from the 75th-percentile QuantReg model on the x-axis.**
*Must Have*
**Proof of Ask:** *"Generate a connected scatter plot for quantile regression diagnostic analysis with x-axis representing fitted values."*

**3. Provides a connected scatter plot for quantile regression diagnostic analysis for Score with residuals computed as observed Score minus fitted values on the y-axis.**
*Must Have*
**Proof of Ask:** *"Generate a connected scatter plot for quantile regression diagnostic analysis with y-axis representing residuals."*

---

### cca_biplot.png

**1. Includes a biplot for CCA canonical variate visualization for NIRF 2024 with the first canonical variate from X variables TLR and GO on the x-axis and the first canonical variate from Y variables RPC and PERCEPTION on the y-axis that is semantically similar to the attached visualization.**
*Must Have*
**Proof of Ask:** *"Generate a biplot for CCA with x-axis representing the first canonical variate derived from TLR and GO and y-axis representing the first canonical variate derived from RPC and PERCEPTION using NIRF 2024 data."*

**2. Provides a biplot for CCA canonical variate visualization for NIRF 2024 with the first canonical variate derived from TLR and GO on the x-axis.**
*Must Have*
**Proof of Ask:** *"Generate a biplot for CCA with x-axis representing the first canonical variate derived from TLR and GO."*

**3. Provides a biplot for CCA canonical variate visualization for NIRF 2024 with the first canonical variate derived from RPC and PERCEPTION on the y-axis.**
*Must Have*
**Proof of Ask:** *"Generate a biplot for CCA with y-axis representing the first canonical variate derived from RPC and PERCEPTION."*

---

### nmf_heatmap.png

**1. Includes a heatmap for NMF latent factor loadings for features TLR, RPC, GO, OI, and PERCEPTION from NIRF 2025 showing the components matrix transposed so that latent factor indices appear on the x-axis and original feature names appear on the y-axis that is semantically similar to the attached visualization.**
*Must Have*
**Proof of Ask:** *"Generate a heatmap for NMF latent factor loadings showing the components matrix transposed with x-axis representing latent factor index and y-axis representing original feature names TLR, RPC, GO, OI, PERCEPTION."*

**2. Provides a heatmap for NMF latent factor loadings for NIRF 2025 displaying the transposed components matrix with latent factor index on the x-axis.**
*Must Have*
**Proof of Ask:** *"Generate a heatmap for NMF latent factor loadings with x-axis representing latent factor index from the transposed components matrix."*

**3. Provides a heatmap for NMF latent factor loadings for NIRF 2025 displaying the transposed components matrix with feature names TLR, RPC, GO, OI, and PERCEPTION on the y-axis.**
*Must Have*
**Proof of Ask:** *"Generate a heatmap for NMF latent factor loadings with y-axis representing original feature names from the transposed components matrix."*

---

### pls_component_contribution.png

**1. Includes a bar plot for PLS component contribution analysis for Score prediction using PLSRegression with n_components equal to 3 on predictors TLR, RPC, GO, OI, and PERCEPTION showing incremental explained variance ratio on the y-axis that is semantically similar to the attached visualization.**
*Must Have*
**Proof of Ask:** *"Generate a bar plot for PLS component contribution analysis with x-axis representing PLS component index and y-axis representing incremental explained variance ratio in target variable Score."*

**2. Provides a bar plot for PLS component contribution analysis for Score prediction with PLS component index from 1 to 3 on the x-axis.**
*Must Have*
**Proof of Ask:** *"Generate a bar plot for PLS component contribution analysis with x-axis representing PLS component index."*

**3. Provides a bar plot for PLS component contribution analysis for Score prediction with incremental explained variance ratio computed from sequential training R-squared differences across PLSRegression components on the y-axis.**
*Must Have*
**Proof of Ask:** *"Generate a bar plot for PLS component contribution analysis with y-axis representing explained variance ratio in target variable Score."*

---

### cdf_score_comparison.png

**1. Includes a cumulative distribution function plot for Score distribution comparison for NIRF 2023 and NIRF 2025 with Score values on the x-axis and cumulative probability on the y-axis that is semantically similar to the attached visualization.**
*Must Have*
**Proof of Ask:** *"Generate a cumulative distribution function plot for Score distribution comparison across NIRF 2023 and 2025 with x-axis representing Score values and y-axis representing cumulative probability."*

**2. Provides a cumulative distribution function plot for Score distribution comparison for NIRF 2023 and 2025 with Score values on the x-axis.**
*Must Have*
**Proof of Ask:** *"Generate a cumulative distribution function plot for Score distribution comparison across years with x-axis representing Score values."*

**3. Provides a cumulative distribution function plot for Score distribution comparison for NIRF 2023 and 2025 with cumulative probability on the y-axis.**
*Must Have*
**Proof of Ask:** *"Generate a cumulative distribution function plot for Score distribution comparison across years with y-axis representing cumulative probability."*

---

### bootstrap_histogram.png

**1. Includes a histogram for bootstrap sampling distribution of mean Score improvement for institutions matched by Institute ID across all three NIRF years with bootstrap sample means on the x-axis and frequency count on the y-axis that is semantically similar to the attached visualization.**
*Must Have*
**Proof of Ask:** *"Generate a histogram for bootstrap sampling distribution of mean Score improvement with x-axis representing bootstrap sample means and y-axis representing frequency count."*

**2. Provides a histogram for bootstrap sampling distribution of mean Score improvement for matched institutions with bootstrap sample means on the x-axis.**
*Must Have*
**Proof of Ask:** *"Generate a histogram for bootstrap sampling distribution of mean Score improvement with x-axis representing bootstrap sample means."*

**3. Provides a histogram for bootstrap sampling distribution of mean Score improvement for matched institutions with frequency count on the y-axis.**
*Must Have*
**Proof of Ask:** *"Generate a histogram for bootstrap sampling distribution of mean Score improvement with y-axis representing frequency count."*

**4. Provides a histogram for bootstrap sampling distribution of mean Score improvement for matched institutions that includes vertical reference lines marking the 2.5th and 97.5th percentiles of the bootstrap means to indicate the 95 percent confidence interval bounds.**
*Must Have*
**Proof of Ask:** *"Include vertical reference lines in the histogram for the 2.5th percentile and 97.5th percentile of bootstrap sample means."*
