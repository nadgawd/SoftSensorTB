export function getDynamicContext(step, ctx = {}) {
  const {
    columns = [],
    rowCount = 0,
    hasTarget = false,
    targetVariable = null,
    selectedFeatures = [],
    hasModel = false,
    r2Score = null,
    hasPlot = false,
  } = ctx

  const numCols = columns.length
  const hasCols = numCols > 0
  const firstCol = hasCols ? columns[0] : null
  const numFeatures = selectedFeatures.length

  switch (step) {
    case 'ingestion':
      return {
        actions: hasCols
          ? [
              'Show me a statistical profile of all columns',
              'Check the data types of all columns',
              `Show me the first 10 rows`,
              hasTarget ? `How many missing values are in ${targetVariable}?` : 'Which column should be my target variable?',
              'Generate a correlation heatmap to see broad relationships',
            ]
          : [
              'What format should my data be in for a soft sensor?',
              'How do I choose a target variable?',
            ],
        tips: hasCols
          ? [
              `✅ Dataset loaded — **${rowCount} rows, ${numCols} columns**.`,
              !hasTarget
                ? `⚡ Click **Set target** on the column you want to predict (e.g., your lab measurement or quality variable).`
                : `🎯 Target set to **${targetVariable}**. Proceed to **Data Quality** to assess sensor health.`,
              '💡 Tip: Ask the assistant to "Profile all columns" for a deep statistical breakdown.',
            ]
          : [
              '📁 Drop a CSV, Parquet, or TXT file to begin.',
              '💡 Your data should have process sensor columns (X) and at least one lab measurement column (y).',
            ],
      }

    case 'quality':
      return {
        actions: hasCols
          ? [
              'Profile columns to get descriptive statistics',
              'Which columns have more than 10% missing data?',
              'Drop columns with more than 50% missing data',
              'Remove missing data using the drop strategy',
              'Detect and remove multivariate anomalies using Isolation Forest',
              'Find and remove outliers using the IQR method',
            ]
          : ['What data quality checks should I run?'],
        tips: hasCols
          ? [
              `📊 **${numCols}** columns available. Use the assistant to investigate data quality.`,
              rowCount < 100
                ? `⚠ Only **${rowCount} rows** detected. Consider gathering more data for robust modeling.`
                : `✅ **${rowCount} rows** available. Good sample size for soft-sensor modeling.`,
              '💡 Missing data or extreme outliers can break downstream models. Ask the assistant to **remove outliers** or **impute missing values**.',
            ]
          : [
              '⬅ Upload a dataset in Step 1 first.',
            ],
      }

    case 'preprocessing':
      return {
        actions: hasCols
          ? [
              'Create a 3-step lag feature for all numeric columns to capture time dynamics',
              'Calculate a rolling mean over a window of 5',
              'Normalize all numeric columns using standard scaler',
              'Apply a log transform to heavily skewed columns',
              hasTarget ? `Plot a time series of ${targetVariable} vs index` : `Plot a time series of ${firstCol}`,
              hasTarget ? `Create a box plot of ${targetVariable} to check distributions` : `Show a violin plot of ${firstCol}`,
            ]
          : ['What preprocessing steps are recommended before modeling?'],
        tips: hasCols
          ? [
              hasPlot
                ? '📈 Chart updated. You can ask for a **trendline** or color it by another column.'
                : '💡 Click a suggested action to generate a chart or transform the data.',
              '🔧 Each preprocessing step creates a new dataset version — you can roll back anytime via the **History** button.',
              rowCount < 50
                ? `⚠ **${rowCount} rows** is very small. Be careful with lagging which drops initial rows.`
                : `✅ **${rowCount} rows** ready for preprocessing.`,
              '💡 Tip: Time-series sensors benefit heavily from **lag features** and **rolling aggregates**.',
            ]
          : ['⬅ Upload a dataset in Step 1 first.'],
      }

    case 'feature-selection':
      return {
        actions: hasCols
          ? [
              hasTarget
                ? `What features are most correlated with ${targetVariable}?`
                : 'Which features correlate most with the target variable?',
              'Run a correlation analysis between all columns',
              'Drop highly collinear features (correlation > 0.95)',
              hasTarget ? `Use RFE (Recursive Feature Elimination) to pick the best 5 features for ${targetVariable}` : 'Select features using RFE',
              'Generate a pair plot (scatter matrix) of the top features',
              numFeatures > 0
                ? `Show a scatter plot of features vs ${targetVariable}`
                : 'Which columns should I use as features for modeling?',
            ]
          : ['How do I select features for a soft sensor?'],
        tips:
          numFeatures > 0
            ? [
                `✅ **${numFeatures} features** selected for modeling.`,
                hasTarget
                  ? `🎯 Target: **${targetVariable}**. Ready to move to Modeling.`
                  : '⚡ Set a target variable in Step 1 before modeling.',
                '💡 More features isn\'t always better — ask the assistant to **Drop collinear features** to improve model stability.',
              ]
            : hasCols
              ? [
                  '⚡ Click columns in the table above to toggle them as features (X).',
                  hasTarget
                    ? `🎯 Target variable is **${targetVariable}** — select the remaining columns as inputs.`
                    : '⚡ Set a target variable first in Step 1.',
                  '💡 Tip: You can ask the AI to automatically run **RFE (Recursive Feature Elimination)**.',
                ]
              : ['⬅ Upload a dataset in Step 1 first.'],
      }

    case 'modeling':
      return {
        actions: hasCols
          ? [
              numFeatures > 0 && hasTarget
                ? `Train a PLS model with 2 components to predict ${targetVariable}`
                : 'Train a PLS soft sensor model',
              numFeatures > 0 && hasTarget
                ? `Train a Ridge regression model to predict ${targetVariable}`
                : 'Train a Ridge regression model',
              hasModel ? 'Retrain using Lasso to force sparsity' : 'What is the difference between PLS and PCR?',
              hasModel ? 'Show the feature coefficients plot' : 'Train a k-NN model with 5 neighbors',
              hasModel
                ? 'Show the variable importance plot (VIP)'
                : numFeatures > 0 && hasTarget
                  ? `Train a linear regression (OLS) model to predict ${targetVariable}`
                  : 'Train a linear regression (OLS) model',
            ]
          : ['What are the different regression algorithms available?'],
        tips: hasModel
          ? [
              r2Score !== null
                ? r2Score > 0.8
                  ? `🏆 **R² = ${r2Score.toFixed(3)}** — Excellent model fit!`
                  : r2Score > 0.6
                    ? `✅ **R² = ${r2Score.toFixed(3)}** — Good model. Consider feature tuning to improve.`
                    : `⚠ **R² = ${r2Score.toFixed(3)}** — Moderate fit. Try more features or a different algorithm.`
                : '📊 Model trained. Check R² and RMSE to assess performance.',
              '💡 Go to Validation (Step 6) to generate parity and residual diagnostic plots.',
            ]
          : hasCols
            ? [
                '🔬 **PLS**, **PCR**, and **Ridge** are highly recommended for process data with correlated inputs.',
                numFeatures === 0 ? '⚡ Select features in Step 4 first, then train here.' : `✅ **${numFeatures} features** ready. Ask the assistant to train a model.`,
              ]
            : ['⬅ Upload a dataset in Step 1 first.'],
      }

    case 'validation':
      return {
        actions: hasModel
          ? [
              'Generate a parity plot (Actual vs Predicted)',
              'Generate a residuals diagnostic plot',
              'Show the feature importance bar chart',
              'Show the signed regression coefficients plot',
              r2Score !== null && r2Score < 0.7
                ? 'How can I improve the model R² score?'
                : 'What are the next steps for deployment?',
            ]
          : ['What is a parity plot in soft sensor validation?', 'What is a residuals plot?'],
        tips: hasModel
          ? [
              '📉 The **Parity Plot** compares model predictions (Y-axis) vs actual lab measurements (X-axis).',
              '🔍 The **Residuals Plot** is crucial for checking if errors are random or if there is a systematic time drift.',
              r2Score !== null
                ? r2Score > 0.85
                  ? '🏆 Strong model performance — suitable for soft-sensor deployment.'
                  : r2Score > 0.65
                    ? '✅ Acceptable performance. Investigate outliers in the parity plot.'
                    : '⚠ Model needs improvement. Consider revisiting feature selection or data quality.'
                : '💡 Ask the assistant to generate a parity or residuals plot to visualize performance.',
            ]
          : [
              '⚡ Train a model in Step 5 first to enable validation.',
              '💡 Validation quantifies how well the model predicts unseen data.',
            ],
      }

    case 'export':
      return {
        actions: [
          'Summarize the entire pipeline and results',
          hasModel ? 'What model coefficients were used?' : 'What steps are still needed before export?',
          'Generate a technical report of the soft sensor development',
          'What deployment considerations should I be aware of?',
        ],
        tips: [
          hasModel && r2Score !== null
            ? r2Score > 0.7
              ? `✅ Model **R² = ${r2Score.toFixed(3)}** — ready for export.`
              : `⚠ Model **R² = ${r2Score.toFixed(3)}** — consider improving before deployment.`
            : '⚡ Complete Modeling and Validation steps before exporting.',
          '📦 The JSON export includes all pipeline decisions for reproducibility.',
        ],
      }

    default:
      return { actions: [], tips: [] }
  }
}
