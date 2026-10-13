export function getSuggestions(step, ctx = {}) {
  const {
    columns = [],
    hasTarget = false,
    targetVariable = null,
    selectedFeatures = [],
    hasModel = false,
    r2Score = null,
  } = ctx

  const hasCols = columns.length > 0
  const firstCol = hasCols ? columns[0] : null
  const numFeatures = selectedFeatures.length

  switch (step) {
    case 'ingestion':
      return hasCols
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
          ]

    case 'quality':
      return hasCols
        ? [
            'Profile columns to get descriptive statistics',
            'Which columns have more than 10% missing data?',
            'Drop columns with more than 50% missing data',
            'Remove missing data using the drop strategy',
            'Detect and remove multivariate anomalies using Isolation Forest',
            'Find and remove outliers using the IQR method',
          ]
        : ['What data quality checks should I run?']

    case 'preprocessing':
      return hasCols
        ? [
            'Create a 3-step lag feature for all numeric columns to capture time dynamics',
            'Calculate a rolling mean over a window of 5',
            'Normalize all numeric columns using standard scaler',
            'Apply a log transform to heavily skewed columns',
            hasTarget ? `Plot a time series of ${targetVariable} vs index` : `Plot a time series of ${firstCol}`,
            hasTarget ? `Create a box plot of ${targetVariable} to check distributions` : `Show a violin plot of ${firstCol}`,
          ]
        : ['What preprocessing steps are recommended before modeling?']

    case 'feature-selection':
      return hasCols
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
        : ['How do I select features for a soft sensor?']

    case 'modeling':
      return hasCols
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
        : ['What are the different regression algorithms available?']

    case 'validation':
      return hasModel
        ? [
            'Generate a parity plot (Actual vs Predicted)',
            'Generate a residuals diagnostic plot',
            'Show the feature importance bar chart',
            'Show the signed regression coefficients plot',
            r2Score !== null && r2Score < 0.7
              ? 'How can I improve the model R² score?'
              : 'What are the next steps for deployment?',
          ]
        : ['What is a parity plot in soft sensor validation?', 'What is a residuals plot?']

    case 'export':
      return [
        'Summarize the entire pipeline and results',
        hasModel ? 'What model coefficients were used?' : 'What steps are still needed before export?',
        'Generate a technical report of the soft sensor development',
        'What deployment considerations should I be aware of?',
      ]

    default:
      return []
  }
}
