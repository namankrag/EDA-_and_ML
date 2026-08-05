==============================================================================
                      LOAN APPROVAL PREDICTION -- README                      
==============================================================================

OVERVIEW
------------------------------------------------------------------------------
This notebook (project.ipynb) trains and compares several binary
classification models that predict whether a loan application will be approved
(target column: Loan_Approved, Yes/No) from applicant demographic, financial,
and loan-related attributes. It walks through the full pipeline: missing-value
handling, exploratory data analysis, encoding, correlation analysis, feature
engineering, and a comparison of six different classification algorithms.

CONTENTS
------------------------------------------------------------------------------
  project.ipynb
      Main notebook (43 cells: 8 markdown, 35 code)
  loan_approval_data.csv
      Source dataset -- NOT included with the notebook. Add it yourself
      in the same folder before running (see DATASET below).

REQUIREMENTS
------------------------------------------------------------------------------
  Python 3.11 (developed on 3.11.9; 3.9+ should also work)

  Packages:
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    kneed

  Install with:
    pip install pandas numpy matplotlib seaborn scikit-learn kneed

DATASET
------------------------------------------------------------------------------
  File:    loan_approval_data.csv
  Rows:    1,000 applicants (950 with a non-missing Loan_Approved label)
  Target:  Loan_Approved (Yes / No) -- moderately imbalanced. Verified counts
           among the 950 labeled rows: 652 No / 298 Yes (~68.6% / 31.4%),
           consistent with the ~70% / 30% split reported from the test-set
           confusion matrices below.

  Columns:
    Applicant_ID        Row identifier (dropped before modeling)
    Applicant_Income    Primary applicant income
    Coapplicant_Income  Co-applicant income
    Employment_Status   Salaried / Self-employed / Unemployed / Contract
    Age                 Applicant age
    Marital_Status      Married / Single
    Dependents          Number of dependents
    Credit_Score        Applicant credit score
    Existing_Loans      Number of existing loans
    DTI_Ratio           Debt-to-income ratio
    Savings             Applicant savings balance
    Collateral_Value    Value of offered collateral
    Loan_Amount         Requested loan amount
    Loan_Term           Loan term, in months
    Loan_Purpose        Personal / Car / Business / Education / Home
    Property_Area       Urban / Semiurban / Rural
    Education_Level     Graduate / Not Graduate
    Gender              Male / Female
    Employer_Category   Private / Government / MNC / Unemployed / Business
    Loan_Approved       TARGET -- Yes / No

  The raw file has missing values scattered across several columns; these
  are imputed in step 2 of the workflow below. Verified against the attached
  loan_approval_data.csv: every one of the 20 columns (including Applicant_ID
  and the Loan_Approved target itself) has exactly 50 missing values (5%),
  but on different rows for each column -- 649 of the 1,000 rows have at
  least one missing field, and no row is missing every field. Because
  Loan_Approved has 50 missing entries, the "1,000 applicants" figure above
  includes 50 rows with no label; only 950 rows have a usable target.

NOTEBOOK WORKFLOW
------------------------------------------------------------------------------
  1. Imports & Load Data
       - Loads loan_approval_data.csv into a pandas DataFrame.

  2. Handling Missing Values
       - Numeric columns -> imputed with the column mean
       - Categorical columns -> imputed with the most frequent value
         (sklearn.impute.SimpleImputer)

  3. Exploratory Data Analysis (EDA)
       - Pie chart of the Loan_Approved class split
       - Boxen and box plots of Applicant_Income, Credit_Score, DTI_Ratio and
         Savings, grouped by Loan_Approved
       - Applicant_ID dropped (identifier only, not predictive)

  4. Encoding
       - Label encoding: Education_Level, Loan_Approved (target)
       - One-hot encoding (drop-first): Employer_Category, Gender,
         Property_Area, Loan_Purpose, Marital_Status, Employment_Status
       - Result: 27 feature columns after encoding (verified: 11 numeric +
         1 label-encoded Education_Level + 15 one-hot dummy columns from
         Employer_Category, Gender, Property_Area, Loan_Purpose,
         Marital_Status and Employment_Status, each dropping its first
         level)

  5. Correlation Heatmap
       - Full correlation matrix, plus features ranked by correlation with
         Loan_Approved

  6. Train/Test Split & Scaling
       - 80/20 split (random_state=42); features standardized with
         StandardScaler

  7. Baseline Models (original features)
       - Logistic Regression
       - K-Nearest Neighbors (k auto-selected 1-8 via elbow method /
         KneeLocator)
       - Gaussian Naive Bayes
       - Best result at this stage: Naive Bayes (highest precision)

  8. Feature Engineering
       - Adds DTI_Ratio_sq and Credit_Score_sq (squared terms)
       - Drops the original DTI_Ratio and Credit_Score
       - Re-splits and re-scales the data

  9. Model Retraining & Comparison (engineered features)
       - Logistic Regression
       - K-Nearest Neighbors
       - Gaussian Naive Bayes
       - Decision Tree: default, pre-pruned (grid search over max_depth /
         min_samples_split), post-pruned (cost-complexity / ccp_alpha search),
         and pre+post combined
       - SVM: default (RBF), kernel comparison (linear / poly / sigmoid), and
         a final tuned polynomial-kernel model

  10. Ensemble Learning
       - Section header only -- the cell is currently empty. Placeholder for
         future work (e.g. Random Forest, Voting or Stacking Classifier).

RESULTS SUMMARY
------------------------------------------------------------------------------
All scores measured on the held-out 20% test set (200 samples).

Baseline models (original features):
  Model                          Precision   Recall       F1  Accuracy
  --------------------------------------------------------------------
  Logistic Regression                0.783    0.770    0.777     0.865
  KNN (k=4)                          0.724    0.344    0.467     0.760
  Naive Bayes                        0.804    0.738    0.769     0.865

After feature engineering (DTI_Ratio_sq, Credit_Score_sq):
  Model                          Precision   Recall       F1  Accuracy
  --------------------------------------------------------------------
  Logistic Regression                0.790    0.803    0.797     0.875
  KNN (k=4)                          0.714    0.328    0.449     0.755
  Naive Bayes                        0.783    0.770    0.777     0.865
  Decision Tree (default)            0.817    0.803    0.810     0.885
  Decision Tree (pre-pruned)         0.824    0.918    0.868     0.915
  Decision Tree (post-pruned)        0.824    0.918    0.868     0.915
  Decision Tree (pre+post)           0.824    0.918    0.868     0.915
  SVM (RBF, default)                 0.774    0.672    0.719     0.840
  SVM (polynomial kernel)            0.848    0.459    0.596     0.810

  Best model: Decision Tree, pre-pruned (max_depth=5, min_samples_split=20)
  -- Accuracy 0.915, F1 0.868, Recall 0.918.

HOW TO RUN
------------------------------------------------------------------------------
  1. Place loan_approval_data.csv in the same folder as project.ipynb (or edit
     the path used in the pd.read_csv() call).
  2. Install the packages listed under REQUIREMENTS.
  3. Open the notebook and run all cells top to bottom:
     jupyter notebook project.ipynb
  4. The final "Ensemble Learning" cell is empty and will not error, but
     produces no output -- it is a placeholder for future work.

NOTES
------------------------------------------------------------------------------
  - Ensemble Learning section is unfinished (see workflow step 10).
  - In the Decision Tree pruning cells, the hyperparameter search loops (the
    min_samples_split loop, and the cost-complexity / ccp_alpha search) fit
    and score against X_train_sc / X_test -- the arrays from BEFORE feature
    engineering -- rather than the updated X_train_scaled / X_test_scaled. The
    final reported models are refit correctly on the engineered data, but it's
    worth rerunning those searches on the post-feature-engineering arrays to
    confirm min_samples_split=20 and the chosen ccp_alpha are still optimal.
  - KNN trails the other models on Recall/F1 in both rounds and may benefit
    from a wider k search or distance weighting.
  - Dataset is modestly sized (1,000 rows) and imbalanced toward "No"; weigh
    the absolute performance numbers with that in mind.
  - Loan_Approved itself has 50 missing values in the raw file. If step 2's
    categorical imputer (most-frequent-value) is applied to the target along
    with the other categorical columns, those 50 rows get auto-labeled "No"
    (the majority class) rather than dropped, which can quietly bias the
    class balance and inflate apparent performance on "No". Worth double-
    checking that the notebook excludes Loan_Approved from that imputer, or
    drops the unlabeled rows instead.
  - The category lists above were cross-checked against the attached CSV:
    Employment_Status has a 4th level, "Contract", and Employer_Category has
    a 5th level, "Business", neither of which was previously documented here.
    Since encoding is one-hot with drop-first, these extra levels are already
    reflected in the 27-feature count and shouldn't require code changes --
    just note them if you inspect dummy-column names.

==============================================================================
