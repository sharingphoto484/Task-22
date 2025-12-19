# Negative Weighted Rubrics: Patient Survival Prediction Analysis

## Failure Ideas Repository: Healthcare Predictive Analytics

### Failure Idea 1: Standardization Variance Bias in Logistic Regression Coefficients

**Failure Idea:** The analyst reports a Patient_Age standardized coefficient of negative 0.1744, which deviates from the correct value of negative 0.1745. This discrepancy arises when the standardization process applies biased variance estimation by setting degrees of freedom to zero in the standard deviation calculation, or when using a different solver algorithm such as lbfgs with varying convergence tolerances. While the difference appears minimal, coefficient precision to four decimal places is critical for reproducible machine learning pipelines where downstream probability calculations and risk stratification depend on exact parameter values. The error suggests inconsistent preprocessing where the StandardScaler may have been configured differently than the standard zero mean unit variance transformation with degrees of freedom equals one, leading to slightly altered feature scales that propagate through gradient descent optimization and produce different final coefficients even with identical random states.

**Corresponding Rubric Item:**
- **Reports that the Patient_Age standardized coefficient is negative 0.1744 instead of negative 0.1745.**
- **-20 points · must have criteria**

---

### Failure Idea 2: Cross-Validation Fold Stratification Inconsistency

**Failure Idea:** The model reports an XGBoost five-fold cross-validated mean AUC of 0.7380 when the correct value is 0.7413, representing a meaningful difference of 0.0033 in discriminative performance. This error typically occurs when the cross-validation splitting does not preserve random state consistency across the XGBoost classifier initialization and the cross-validation splitter itself, or when stratified k-fold splitting fails to maintain proportional class distributions in each fold. In healthcare prediction contexts where survival outcomes may have imbalanced class distributions with approximately sixty-three percent survivors and thirty-seven percent non-survivors, improper stratification can create folds with varying baseline survival rates that inflate or deflate performance metrics. The discrepancy suggests that the cross-validation procedure either used a different random seed, applied non-stratified splitting, or employed a different version of the XGBoost library where minor algorithmic updates between versions 1.x and 2.x alter tree construction heuristics even under identical hyperparameters.

**Corresponding Rubric Item:**
- **Reports that the XGBoost mean cross-validated AUC is 0.7380 instead of 0.7413.**
- **-25 points · must have criteria**

---

### Failure Idea 3: Test Set Leakage Through Improper Standardization

**Failure Idea:** The analyst reports an SVM mean predicted survival probability of 0.6346 on the Testing_set_advance dataset, substantially higher than the correct value of 0.6209, indicating a systematic upward bias in probability predictions. This error arises when test features are standardized using statistics computed from the test set itself rather than applying the training set mean and standard deviation for transformation. Such test set leakage violates the fundamental principle that test data must remain unseen during all preprocessing steps, and allows the model to exploit distributional information from the test population that would be unavailable in true deployment scenarios. The elevated predictions suggest the model fitted to a feature space that incorporated test set characteristics, artificially improving apparent survival probabilities by approximately two percentage points and creating overly optimistic estimates that would not generalize to new patient populations.

**Corresponding Rubric Item:**
- **Reports that the SVM mean predicted survival probability is 0.6346 instead of 0.6209.**
- **-25 points · must have criteria**

---

### Failure Idea 4: Feature Encoding Artifacts in KNN Classification

**Failure Idea:** The model reports a KNN predicted positive survival rate of 69.17 percent on Testing_set_intermediate when the correct value is 66.24 percent, representing an overestimation of nearly three percentage points in the proportion of patients predicted to survive. This discrepancy emerges when the Diagnosed_Condition categorical feature is included without proper encoding or when different missing value handling strategies retain more test records than the specified complete case analysis. KNN algorithms are particularly sensitive to feature scaling and categorical variable treatment because distance calculations in Euclidean space treat all numeric values as continuous measurements. If Diagnosed_Condition was included as raw integer codes without one-hot encoding or if different standardization approaches such as min-max normalization were applied instead of zero mean unit variance scaling, the distance calculations between neighbors would be distorted, causing the classifier to identify different nearest neighbors and produce more optimistic survival predictions.

**Corresponding Rubric Item:**
- **Reports that the KNN predicted positive survival rate is 69.17 percent instead of 66.24 percent.**
- **-20 points · must have criteria**

---

### Failure Idea 5: Train-Validation Split Randomization Error

**Failure Idea:** The analyst reports a validation AUC score of 0.5973 from ROC curve analysis when the correct value is 0.6200, representing a substantial underestimation of model discrimination ability by 0.0227 AUC points. This error occurs when the train-validation split uses a different random state than the specified forty-two or fails to set any random state, causing different patient records to be allocated to training versus validation sets. In medical prediction tasks, patient heterogeneity means that random samples can have different baseline survival rates, comorbidity distributions, and demographic compositions. A validation set that happens to contain disproportionately more complex cases with mixed survival predictors will yield lower AUC scores even for a properly trained model. The magnitude of this discrepancy suggests either a fundamentally different data split or that validation set features were preprocessed using validation statistics rather than training statistics, both of which would degrade apparent model performance and provide misleading evidence about the logistic regression classifier's ability to discriminate between survival outcomes.

**Corresponding Rubric Item:**
- **Reports that the validation AUC score is 0.5973 instead of 0.6200.**
- **-25 points · must have criteria**

---

### Failure Idea 6: Confusion of Feature Importance with Causal Impact

**Failure Idea:** An analyst might interpret the random forest identification of Patient_Body_Mass_Index as the most important feature with an importance score of 0.4062 as evidence that BMI causally determines one-year survival outcomes and that clinical interventions targeting BMI reduction would directly improve survival probabilities. This reasoning conflates statistical association measured through mean decrease in impurity with causal mechanisms. Feature importance scores quantify how much a variable contributes to reducing prediction uncertainty in the training data, but this reflects correlational patterns that may arise from confounding variables, reverse causation, or shared upstream determinants. For instance, low BMI might correlate with survival not because body mass directly causes mortality but because underlying conditions such as cachexia, chronic disease progression, or frailty simultaneously reduce both BMI and survival probability. The random forest has no mechanism to distinguish between BMI as a causal driver versus BMI as a symptom marker, and the importance score only indicates that BMI partitions patients into groups with different observed survival rates in the historical dataset.

**Corresponding Rubric Item:**
- **States that Patient_Body_Mass_Index causally determines one-year survival outcomes based on its random forest importance score.**
- **-12 points · nice to have criteria**

---

### Failure Idea 7: Sample Period Generalization Fallacy

**Failure Idea:** An analyst might conclude that the XGBoost model with a mean cross-validated AUC of 0.7413 establishes a reliable and permanent predictive framework for patient survival that will maintain this discrimination ability across all future patient populations and care settings. This reasoning treats sample-based cross-validation performance as evidence of invariant predictive validity, when in fact the AUC score is conditional on the specific patient population, treatment protocols, diagnostic practices, and healthcare system characteristics present during the 25,079-record training period. Medical prediction models are known to experience performance degradation when applied to different hospitals, geographic regions, time periods, or patient demographics due to distribution shift, changing treatment standards, evolving disease prevalence, and differing care quality. The 0.7413 AUC represents in-sample discriminative ability under five-fold cross-validation but does not guarantee that the same model architecture and hyperparameters will achieve comparable performance when deployed prospectively or in external validation cohorts.

**Corresponding Rubric Item:**
- **Claims that the XGBoost cross-validated AUC of 0.7413 establishes permanent predictive validity across all future patient populations and care settings.**
- **-12 points · nice to have criteria**

---

### Failure Idea 8: Probability Calibration Misinterpretation

**Failure Idea:** An analyst might interpret the SVM mean predicted survival probability of 0.6209 on the Testing_set_advance dataset as indicating that the average patient in this cohort has a sixty-two percent chance of one-year survival and use this probability directly for clinical decision-making or resource allocation. This reasoning assumes that SVM predicted probabilities are well-calibrated and represent true event frequencies, when in fact support vector machines with RBF kernels produce probability estimates through Platt scaling that may be systematically biased. The mean predicted probability aggregates individual patient predictions but does not necessarily correspond to the actual survival rate that would be observed if these 9,330 patients were followed prospectively. Calibration analysis comparing predicted probabilities to observed outcomes across probability bins would be required to assess whether 0.6209 represents a genuine survival estimate or merely a model output that ranks patients by risk without accurate probability quantification.

**Corresponding Rubric Item:**
- **Interprets the SVM mean predicted probability of 0.6209 as the true expected survival rate for the Testing_set_advance population suitable for direct clinical decision-making.**
- **-10 points · nice to have criteria**

---

### Failure Idea 9: Algorithmic Performance as Evidence of Optimal Strategy

**Failure Idea:** An analyst might assert that because XGBoost achieved a higher cross-validated AUC of 0.7413 compared to the logistic regression validation AUC of 0.6200, gradient boosting is definitively superior to logistic regression for this prediction task and should be exclusively deployed without considering ensemble combinations or probability calibration. This reasoning treats a single performance metric on one validation approach as comprehensive evidence of model superiority, ignoring that different algorithms may excel on different performance dimensions such as calibration, interpretability, computational efficiency, or robustness to distribution shift. The comparison also conflates different validation strategies where five-fold cross-validation for XGBoost uses eighty percent of data for training in each fold while the ROC analysis uses a single eighty-twenty split, making direct AUC comparisons potentially misleading. Furthermore, the statement neglects that optimal prediction strategies often combine multiple algorithms through stacking or blending to leverage complementary strengths rather than relying on a single best-performing model.

**Corresponding Rubric Item:**
- **Concludes that XGBoost is definitively superior to logistic regression based solely on comparing cross-validated AUC of 0.7413 to validation AUC of 0.6200 without considering calibration, interpretability, or ensemble strategies.**
- **-10 points · nice to have criteria**

---

### Failure Idea 10: Demographic Imbalance as Sampling Bias Indicator

**Failure Idea:** An analyst might interpret the smoking status count ratio of 263.27 in the Testing_set_advance dataset as evidence of severe sampling bias or data collection error that invalidates the entire test set for model evaluation purposes. This reasoning assumes that extreme category imbalance necessarily reflects methodological problems rather than true population characteristics. However, the ratio calculation dividing maximum category count 3,949 by minimum category count 15 captures the relationship between the most and least prevalent smoking status categories, which may legitimately reflect that very few patients fall into rare categories such as ambiguous smoking status codes or data entry variants like "YESS" versus "YES". The imbalance does not inherently compromise the test set validity if it accurately represents the target patient population distribution, and the model predictions should be evaluated on the actual patient mix encountered in clinical practice rather than artificially balanced samples that would misrepresent real-world deployment conditions.

**Corresponding Rubric Item:**
- **Asserts that the smoking status count ratio of 263.27 indicates severe sampling bias that invalidates the Testing_set_advance dataset for model evaluation.**
- **-8 points · nice to have criteria**

---

### Failure Idea 11: Near-Zero Correlation as Independence Proof

**Failure Idea:** An analyst might conclude that the Pearson correlation of 0.0032 between Patient_Age and Patient_Body_Mass_Index demonstrates complete statistical independence between these demographic variables, implying that age provides no information about BMI and vice versa, and therefore both variables contribute entirely orthogonal information to survival prediction models. This reasoning conflates linear correlation with general statistical independence and assumes that near-zero Pearson correlation establishes the absence of any relationship. However, Pearson correlation only measures linear association, and age and BMI could have nonlinear relationships, threshold effects, or age-specific BMI patterns that are not captured by a single correlation coefficient. For example, BMI trajectories may differ between young and elderly patients, or certain age groups may have restricted BMI ranges due to physiological factors. The 0.0032 correlation indicates no strong linear trend across the entire population but does not prove that age and BMI are unrelated in all subgroups or that controlling for one variable would have no effect on the relationship between the other and survival.

**Corresponding Rubric Item:**
- **Claims that the Pearson correlation of 0.0032 proves complete statistical independence between Patient_Age and Patient_Body_Mass_Index across all patient subgroups.**
- **-8 points · nice to have criteria**

---

### Failure Idea 12: Age Difference as Population Drift Evidence

**Failure Idea:** An analyst might interpret the absolute age difference of 1.19 years between training survivors with mean age 32.06 and Testing_set_intermediate with mean age 33.25 as evidence of significant population drift requiring model recalibration or retraining before deployment. This reasoning treats a modest difference in mean age as indicative of fundamentally different patient populations when in fact 1.19 years represents a negligible shift relative to the full age range in healthcare datasets that typically span from young adults to elderly patients across six or seven decades. The comparison also conflates training survivors specifically with all testing patients generally, mixing conditional and marginal distributions. The training survivor mean reflects patients who lived one year conditional on their characteristics, while the testing mean includes all patients regardless of unknown survival status. This 1.19-year difference could simply reflect natural sampling variation or slightly different age distributions between datasets without implying that the model trained on one population would fail to generalize to the other.

**Corresponding Rubric Item:**
- **Asserts that the age difference of 1.19 years indicates significant population drift requiring model recalibration before deployment to Testing_set_intermediate.**
- **-8 points · nice to have criteria**

---

### Failure Idea 13: Negative Age Coefficient as Universal Mortality Predictor

**Failure Idea:** An analyst might interpret the negative Patient_Age coefficient of negative 0.1745 in the logistic regression model as establishing that increasing age universally decreases survival probability across all patient subgroups and clinical contexts, and therefore age-based risk stratification should prioritize older patients for aggressive interventions. This reasoning treats a population-level coefficient as evidence of individual-level causal effects and assumes the age-survival relationship is monotonic and homogeneous. However, the negative coefficient represents the average association between standardized age and log-odds of survival after controlling for BMI and six binary health indicators, but this relationship may vary substantially across subpopulations defined by comorbidities, treatment types, or disease stages. For certain conditions, younger patients might have worse prognoses due to more aggressive disease presentations, or the age effect might be nonlinear with highest mortality in both very young and very old patients. The coefficient provides a summary statistic for the training sample under a linear additive model structure but does not prove that age has the same directional impact for every patient or that chronological age mechanistically causes mortality independent of associated health deterioration.

**Corresponding Rubric Item:**
- **States that the negative Patient_Age coefficient of negative 0.1745 establishes that increasing age universally decreases survival probability across all patient subgroups and clinical contexts.**
- **-10 points · nice to have criteria**

---

### Failure Idea 14: Validation AUC as Deployment Performance Guarantee

**Failure Idea:** An analyst might conclude that the validation AUC score of 0.6200 from the logistic regression ROC analysis guarantees that the deployed model will maintain this exact discrimination ability when applied to new patient populations in clinical practice. This reasoning treats a single validation set performance metric as a fixed model characteristic rather than recognizing that AUC scores are estimates subject to sampling variability and population dependence. The 0.6200 AUC was computed on a twenty percent validation split from the Training_set_advance dataset using random state forty-two, representing one particular realization of the train-validation partition. Different random splits would yield different AUC values due to sampling variation, and the validation set itself may not be representative of future patient populations that could have different demographic distributions, disease prevalence, treatment patterns, or healthcare access. The validation AUC provides a point estimate of discrimination ability on held-out historical data but requires confidence intervals, external validation, and prospective evaluation to assess whether 0.6200 represents robust predictive performance or an optimistic estimate specific to this particular data split.

**Corresponding Rubric Item:**
- **Claims that the validation AUC of 0.6200 guarantees the deployed model will maintain this exact discrimination ability when applied to new patient populations.**
- **-10 points · nice to have criteria**

---

### Failure Idea 15: KNN Predictions as True Survival Proportions

**Failure Idea:** An analyst might interpret the KNN predicted positive rate of 66.24 percent on Testing_set_intermediate as establishing that exactly sixty-six percent of patients in this test cohort will survive one year if they receive standard care protocols. This reasoning conflates model predictions with ground truth outcomes and assumes that the proportion of positive predictions directly corresponds to the actual survival rate that would be observed empirically. However, KNN classifiers produce discrete class assignments based on majority voting among five nearest neighbors using Euclidean distance in standardized feature space, and the 66.24 percent positive rate represents the percentage of test patients whose five nearest training neighbors had predominantly positive survival outcomes. This algorithmic output depends on the specific training data composition, feature scaling, distance metric choice, and k parameter setting, none of which guarantee that predicted class proportions match true outcome frequencies. Calibration analysis would be required to assess whether sixty-six percent predicted survival corresponds to a genuine sixty-six percent observed survival rate or whether the KNN classifier is systematically over-predicting or under-predicting survival.

**Corresponding Rubric Item:**
- **Interprets the KNN predicted positive rate of 66.24 percent as the true proportion of Testing_set_intermediate patients who will survive one year.**
- **-10 points · nice to have criteria**

---

## Negative Weighted Rubric Criteria

### Must-Have Criteria (Numerical/Factual Errors)

**1. Reports that the Patient_Age standardized coefficient is negative 0.1744 instead of negative 0.1745 rounded to four decimal places.**
- **-20 points · must have criteria**
- **Rationale:** The prompt explicitly requires reporting the standardized coefficient rounded to four decimal places. This numerical discrepancy of 0.0001 indicates improper standardization preprocessing, solver configuration differences, or premature rounding that violates the specified precision requirement for reproducible machine learning pipelines.

**2. Reports that the XGBoost mean cross-validated AUC is 0.7380 instead of 0.7413 rounded to four decimal places.**
- **-25 points · must have criteria**
- **Rationale:** The prompt requires reporting mean cross-validated AUC score rounded to four decimal places. This 0.0033 difference represents a meaningful discrepancy in model discrimination performance resulting from inconsistent random state settings, non-stratified cross-validation, or XGBoost version differences that compromise reproducibility.

**3. Reports that the SVM mean predicted survival probability is 0.6346 instead of 0.6209 rounded to four decimal places.**
- **-25 points · must have criteria**
- **Rationale:** The prompt requires calculating mean predicted survival probability rounded to four decimal places using training set standardization statistics. The 0.0137 upward bias indicates test set leakage through improper feature scaling or different probability calibration methods that violate fundamental train-test separation principles.

**4. Reports that the KNN predicted positive survival rate is 69.17 percent instead of 66.24 percent rounded to two decimal places.**
- **-20 points · must have criteria**
- **Rationale:** The prompt requires reporting predicted positive rate as percentage rounded to two decimal places. The 2.93 percentage point overestimation suggests improper categorical encoding, inconsistent standardization approaches, or incomplete case filtering that alters distance calculations and neighbor identification.

**5. Reports that the validation AUC score is 0.5973 instead of 0.6200 rounded to four decimal places.**
- **-25 points · must have criteria**
- **Rationale:** The prompt requires reporting validation AUC score rounded to four decimal places using an eighty-twenty split with random state forty-two. The 0.0227 underestimation indicates incorrect train-validation splitting, improper random state configuration, or validation set preprocessing errors that misrepresent model discrimination ability.

---

### Nice-to-Have Criteria (Interpretive/Reasoning Errors)

**6. States that Patient_Body_Mass_Index causally determines one-year survival outcomes based on its random forest importance score of 0.4062.**
- **-12 points · nice to have criteria**
- **Rationale:** Corresponds to Failure Idea 6. Feature importance measures statistical association through mean decrease in impurity but does not establish causal relationships. Interpreting importance scores as causal impact conflates correlational patterns with mechanistic effects.

**7. Claims that the XGBoost cross-validated AUC of 0.7413 establishes permanent predictive validity across all future patient populations and care settings.**
- **-12 points · nice to have criteria**
- **Rationale:** Corresponds to Failure Idea 7. Sample-based cross-validation performance is conditional on the training population characteristics and does not guarantee invariant discrimination ability across different hospitals, time periods, or patient demographics subject to distribution shift.

**8. Interprets the SVM mean predicted probability of 0.6209 as the true expected survival rate for the Testing_set_advance population suitable for direct clinical decision-making.**
- **-10 points · nice to have criteria**
- **Rationale:** Corresponds to Failure Idea 8. SVM probability estimates from Platt scaling may be systematically miscalibrated and do not necessarily represent true event frequencies without calibration validation comparing predicted probabilities to observed outcomes.

**9. Concludes that XGBoost is definitively superior to logistic regression based solely on comparing cross-validated AUC of 0.7413 to validation AUC of 0.6200 without considering calibration, interpretability, or ensemble strategies.**
- **-10 points · nice to have criteria**
- **Rationale:** Corresponds to Failure Idea 9. Single-metric comparisons across different validation approaches ignore complementary model strengths and the potential benefits of ensemble combinations that leverage multiple algorithms.

**10. Asserts that the smoking status count ratio of 263.27 indicates severe sampling bias that invalidates the Testing_set_advance dataset for model evaluation.**
- **-8 points · nice to have criteria**
- **Rationale:** Corresponds to Failure Idea 10. Extreme category imbalance may reflect true population distributions rather than methodological problems, and models should be evaluated on realistic patient mixes rather than artificially balanced samples.

**11. Claims that the Pearson correlation of 0.0032 proves complete statistical independence between Patient_Age and Patient_Body_Mass_Index across all patient subgroups.**
- **-8 points · nice to have criteria**
- **Rationale:** Corresponds to Failure Idea 11. Near-zero linear correlation does not establish general independence and ignores potential nonlinear relationships, threshold effects, or age-specific BMI patterns not captured by Pearson correlation.

**12. Asserts that the age difference of 1.19 years indicates significant population drift requiring model recalibration before deployment to Testing_set_intermediate.**
- **-8 points · nice to have criteria**
- **Rationale:** Corresponds to Failure Idea 12. A 1.19-year mean age difference represents negligible shift relative to typical healthcare dataset age ranges and conflates training survivors with all testing patients without implying generalization failure.

**13. States that the negative Patient_Age coefficient of negative 0.1745 establishes that increasing age universally decreases survival probability across all patient subgroups and clinical contexts.**
- **-10 points · nice to have criteria**
- **Rationale:** Corresponds to Failure Idea 13. Population-level coefficients represent average associations but do not prove homogeneous individual-level effects or account for nonlinear age relationships and subgroup heterogeneity.

**14. Claims that the validation AUC of 0.6200 guarantees the deployed model will maintain this exact discrimination ability when applied to new patient populations.**
- **-10 points · nice to have criteria**
- **Rationale:** Corresponds to Failure Idea 14. Validation AUC is a point estimate subject to sampling variability and population dependence, requiring confidence intervals and external validation to assess robust predictive performance.

**15. Interprets the KNN predicted positive rate of 66.24 percent as the true proportion of Testing_set_intermediate patients who will survive one year.**
- **-10 points · nice to have criteria**
- **Rationale:** Corresponds to Failure Idea 15. Model predictions represent algorithmic outputs based on training neighbor composition and do not directly correspond to true outcome frequencies without calibration validation.

---

## Mapping: Rubric Criteria → Failure Ideas

| Rubric # | Type | Points | Failure Idea Title |
|----------|------|--------|-------------------|
| 1 | Must-have | -20 | Standardization Variance Bias in Logistic Regression Coefficients |
| 2 | Must-have | -25 | Cross-Validation Fold Stratification Inconsistency |
| 3 | Must-have | -25 | Test Set Leakage Through Improper Standardization |
| 4 | Must-have | -20 | Feature Encoding Artifacts in KNN Classification |
| 5 | Must-have | -25 | Train-Validation Split Randomization Error |
| 6 | Nice-to-have | -12 | Confusion of Feature Importance with Causal Impact |
| 7 | Nice-to-have | -12 | Sample Period Generalization Fallacy |
| 8 | Nice-to-have | -10 | Probability Calibration Misinterpretation |
| 9 | Nice-to-have | -10 | Algorithmic Performance as Evidence of Optimal Strategy |
| 10 | Nice-to-have | -8 | Demographic Imbalance as Sampling Bias Indicator |
| 11 | Nice-to-have | -8 | Near-Zero Correlation as Independence Proof |
| 12 | Nice-to-have | -8 | Age Difference as Population Drift Evidence |
| 13 | Nice-to-have | -10 | Negative Age Coefficient as Universal Mortality Predictor |
| 14 | Nice-to-have | -10 | Validation AUC as Deployment Performance Guarantee |
| 15 | Nice-to-have | -10 | KNN Predictions as True Survival Proportions |

---

## Summary

- **Total Failure Ideas:** 15
- **Total Negative Criteria:** 15
- **Must-Have Criteria:** 5 (targeting explicit numerical errors)
- **Nice-to-Have Criteria:** 10 (targeting interpretive/reasoning errors)
- **Maximum Possible Deduction:** -213 points
- **Must-Have Deductions:** -115 points
- **Nice-to-Have Deductions:** -98 points

---

## Coverage Analysis

### Numerical Errors Identified:
1. ✓ Patient_Age coefficient error (0.0001 deviation)
2. ✓ XGBoost CV AUC error (0.0033 deviation)
3. ✓ SVM probability error (0.0137 deviation)
4. ✓ KNN positive rate error (2.93 percentage point deviation)
5. ✓ Validation AUC error (0.0227 deviation)

### Interpretive Error Categories:
1. ✓ Causal inference from correlational measures
2. ✓ Generalization beyond sample validity
3. ✓ Probability calibration assumptions
4. ✓ Single-metric model comparisons
5. ✓ Data quality misinterpretations
6. ✓ Statistical independence conflation
7. ✓ Population drift overstatement
8. ✓ Coefficient universality assumptions
9. ✓ Performance guarantee claims
10. ✓ Prediction-outcome equivalence

### Alignment with Positive Rubrics:
- All must-have negative criteria correspond to explicit numerical requirements in positive rubrics
- Negative weights are proportional to but do not exceed corresponding positive criterion weights
- Nice-to-have criteria penalize unrequested interpretive content that introduces analytical errors
- Criteria are atomic, self-contained, and verifiable from model response text alone
