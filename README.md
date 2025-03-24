# Cancer Classification Project

This repository demonstrates a full machine learning workflow to classify breast tumor biopsy data as benign (B) or malignant (M), using the `dslabs::brca` dataset.

## Steps

1. **Data & EDA**  
   - We look at the data dimensions, class distribution (about 62.7% benign, 37.3% malignant).
   - Examine the means and SDs of each feature.

2. **Scaling**  
   - We subtract each column’s mean and divide by its SD, preparing for PCA and many ML models.

3. **PCA**  
   - We compute principal components (PCs).
   - Our first PC typically explains ~44% of the variance, and ~7 PCs explain over 90%.

4. **Train/Test Split**  
   - We partition the data so 80% goes to a training set, 20% to a test set.

5. **Model Training**  
   We train four models:
   - **Logistic Regression** (accuracy ~93.9% in this run)
   - **Loess** (“gamLoess”) model
   - **k-Nearest Neighbors** (k from 3..21)  
   - **Random Forest** (mtry from 3,5,7,9)

6. **Ensemble**  
   - We form a simple majority vote from the four models’ predictions.

## Results

A typical run yields results like:

| Model                | Accuracy |
|----------------------|---------:|
| Logistic Regression  | 0.939    |
| Loess (gamLoess)     | 0.930    |
| kNN                  | 0.974    |
| Random Forest        | 0.948    |
| Ensemble            | 0.626    |

*(Exact numbers can vary slightly each run.)*

The **kNN** model typically performed the best in this demonstration, with an accuracy of about **97.4%**.

**Random Forest** performed well too (~94.8%), and reported the **top variable** (for malignant classification) as *“area_worst”* in the example run.

Interestingly, the **ensemble** here had a lower accuracy than the best single model. This can happen if one model strongly outperforms the others.

## Usage

1. Clone this repository.  
2. Open the `cancer_classification.R` file in R or RStudio.  
3. Run line by line, or do `source("cancer_classification.R")`.  
4. Check the console output for the final model accuracies and top random forest variable.

## Dependencies

- R >= 4.0
- Packages:
  - `dslabs` (for the `brca` dataset)
  - `matrixStats`
  - `tidyverse` (for data wrangling & ggplot2)
  - `caret` (training ML models)
  - `gam` (for loess method)

## License

This code is provided for educational and illustrative purposes.
