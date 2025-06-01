# Title

Balancing the Scales: A Comprehensive Study on Tackling Class Imbalance in Binary Classification

# Tags

accuracy
auc-score
binary classification
class imbalance
class weights
comparative study
decision threshold
f1-score
f2-score
imbalanced data
pr-auc
precision
recall
smote

![imbalanced-balance-scale-stretched.webp](imbalanced-balance-scale-stretched.webp)

TL;DR  
 This study evaluates three strategies for handling imbalanced datasets in binary classification—SMOTE, class weights, and decision threshold calibration—across 15 classifiers and 30 datasets. Results from 9,000 experiments show all methods generally outperform the baseline, with decision threshold calibration emerging as the most consistent performer. However, significant variability across datasets emphasizes the importance of testing multiple approaches for specific problems.

---

# Abstract

Class imbalance in binary classification tasks remains a significant challenge in machine learning, often resulting in poor performance on minority classes. This study comprehensively evaluates three widely-used strategies for handling class imbalance: Synthetic Minority Over-sampling Technique (SMOTE), Class Weights tuning, and Decision Threshold Calibration. We compare these methods against a baseline scenario across 15 diverse machine learning models and 30 datasets from various domains, conducting a total of 9,000 experiments. Performance was primarily assessed using the F1-score, with additional 9 metrics including F2-score, precision, recall, Brier-score, PR-AUC, and AUC. Our results indicate that all three strategies generally outperform the baseline, with Decision Threshold Calibration emerging as the most consistently effective technique. However, we observed substantial variability in the best-performing method across datasets, highlighting the importance of testing multiple approaches for specific problems. This study provides valuable insights for practitioners dealing with imbalanced datasets and emphasizes the need for dataset-specific analysis in evaluating class imbalance handling techniques.

# Introduction

Binary classification tasks frequently encounter imbalanced datasets, where one class significantly outnumbers the other. This imbalance can severely impact model performance, often resulting in classifiers that excel at identifying the majority class but perform poorly on the critical minority class. In fields such as fraud detection, disease diagnosis, and rare event prediction, this bias can have serious consequences.

One of the most influential techniques developed to address this challenge is the **Synthetic Minority Over-sampling Technique (SMOTE)**, a method proposed by Chawla et al. (2002) that generates synthetic examples of the minority class. Since its introduction, the SMOTE paper has become one of the most cited papers in the field of imbalanced learning, with over 30,000 citations. SMOTE's popularity has spurred the creation of many other oversampling techniques and numerous SMOTE variants. For example, Kovács (2019) documents 85 SMOTE-variants implemented in Python, including:

- **Borderline-SMOTE** (Han et al., 2005)
- **Safe-Level-SMOTE** (Bunkhumpornpat et al., 2009)
- **SMOTE + Tomek** and **SMOTE + ENN** (Batista et al., 2004)

Despite its widespread use, recent studies have raised some criticisms of SMOTE. For instance, Blagus and Lusa (2013) indicate limitations in handling high-dimensional data, while Elor and Averbuch-Elor (2022) and Hulse et al. (2007) suggest the presence of better alternatives for handling class imbalance. This highlights that while SMOTE is a powerful tool, it is not without limitations.

In this study, we aim to provide a more balanced view of techniques for handling class imbalance by evaluating not only SMOTE but also other widely-used strategies, such as Class Weights and Decision Threshold Calibration. These three treatment scenarios target class imbalance at different stages of the machine learning pipeline:

1.  **SMOTE**: Generating synthetic examples of the minority class during data preprocessing.
2.  **Class Weights**: Adjusting the importance of classes during model training.
3.  **Decision Threshold Calibration**: Adjusting the classification threshold post-training.

We compare these strategies with the **Baseline** approach (standard model training without addressing imbalance) to assess their effectiveness in improving model performance on imbalanced datasets. Our goal is to provide insights into which treatment methods offer the most significant improvements in performance metrics such as F1-score, F2-score, accuracy, precision, recall, MCC, Brier score, Matthews Correlation Coefficient (MCC), PR-AUC, and AUC. We also aim to evaluate these techniques across a wide range of datasets and models to provide a more generalizable understanding of their effectiveness.

To ensure a comprehensive evaluation, this study encompasses:

- **30 datasets** from various domains, with sample sizes ranging from ~500 to 20,000 and rare class percentages between 1% and 20%.
- **15 classifier models**, including tree-based methods, boosting algorithms, neural networks, and traditional classifiers.
- Evaluation using 5-fold cross-validation.

In total, we conduct 9,000 experiments involving the 4 scenarios, 15 models, 30 datasets, and validation folds. This extensive approach allows us to compare these methods and their impact on model performance across a wide range of scenarios and algorithmic approaches. It provides a robust foundation for understanding the effectiveness of different imbalance handling strategies in binary classification tasks.

# Methodology

## Datasets

We selected 30 datasets based on the following criteria:

- Binary classification problems
- Imbalanced class distribution (minority class < 20%)
- Sample size ≤ 20,000
- Feature count ≤ 100
- Real-world data from diverse domains
- Publicly available

The characteristics of the selected datasets are summarized in the chart below:

![datasets-summary.png](datasets-summary.png)

The dataset selection criteria were carefully chosen to ensure a comprehensive and practical study:

- The 20% minority class threshold for class imbalance, while somewhat arbitrary, represents a reasonable cut-off point that is indicative of significant imbalance.
- The limitations on sample size (≤ 20,000) and feature count (≤ 100) were set to accommodate a wide range of real-world datasets while ensuring manageable computational resources for an experiment of our scale. This balance allows us to include diverse, practically relevant datasets without compromising the breadth of our study.
- The focus on diverse domains ensures that our models are tested across a wide range of industries and data characteristics, enhancing the generalizability of our findings.

:::info{title="Info"}

 <h2> Dataset Repository </h2>
 
 You can find the study datasets and information about their sources and specific characteristics in the following repository: 
 
 [Imbalanced Classification Study Datasets](https://github.com/readytensor/rt-binary-imbalance-datasets)
 
 This repository is also linked in the **Datasets** section of this publication.
 :::
 
 
 ## Models
 
 Our study employed a diverse set of 15 classifier models, encompassing a wide spectrum of algorithmic approaches and complexities. This selection ranges from simple baselines to advanced ensemble methods and neural networks, including tree-based models and various boosting algorithms. The diversity in our model selection allows us to assess how different imbalanced data handling techniques perform across various model types and complexities.
 
 The following chart lists the models used in our experiments:
 
 
 
 ![classifiers.png](classifiers.png)
 
 A key consideration in our model selection process was ensuring that all four scenarios (Baseline, SMOTE, Class Weights, and Decision Threshold Calibration) could be applied consistently to each model. This criterion influenced our choices, leading to the exclusion of certain algorithms such as k-Nearest Neighbors (KNN) and Naive Bayes Classifiers, which do not inherently support the application of class weights. This careful selection process allowed us to maintain consistency across all scenarios while still representing a broad spectrum of machine learning approaches.

 <h2> Implementation Details </h2>
 Each model is implemented in a separate repository to accommodate differing dependencies, but all are designed to work with any dataset in a generalized manner. These repositories include:
 
 - Training and testing code
 - Docker containerization for environment-independent usage
 - Hyperparameter tuning code, where applicable
 
 To ensure a fair comparison, we used the same preprocessing pipeline for all 15 models and scenarios. This pipeline includes steps such as one-hot encoding, standard scaling, and missing data imputation. The only difference in preprocessing occurs in the SMOTE scenario, where synthetic minority class examples are generated. Otherwise, the preprocessing steps are identical across all models and scenarios, ensuring that the only difference is the algorithm and the specific imbalance handling technique applied.
 
 Additionally, each model's hyperparameters were kept constant across the Baseline, SMOTE, Class Weights, and Decision Threshold scenarios to ensure fair comparisons.
 
 The imbalanced data handling scenarios are implemented in a branch named `imbalance`. A configuration file, `model_config.json`, allows users to specify which scenario to run: `baseline`, `smote`, `class_weights`, or `decision_threshold`.
 
 :::info{title="Info"}
 <h2> Model Repositories </h2>
 
 All model implementations are available in our public repositories, linked in the **Models** section of this publication.
 :::
 
 ## Evaluation Metrics
 
 To comprehensively evaluate the performance of the models across different imbalanced data handling techniques, we tracked the following 10 metrics:
 
 
 
 ![evaluation-metrics.png](evaluation-metrics.png)
 
 
 Our primary focus is on the **F1-score**, a label metric that uses predicted classes rather than underlying probabilities. The F1-score provides a balanced measure of precision and recall, making it particularly useful for assessing performance on imbalanced datasets.
 
 While real-world applications often employ domain-specific cost matrices to create custom metrics, our study spans 30 diverse datasets. The F1-score allows us to evaluate all four scenarios, including decision threshold tuning, consistently across this varied set of problems.
 
 Although our analysis emphasizes the F1-score, we report results for all 10 metrics. Readers can find comprehensive information on model performance across all metrics and scenarios in the detailed results repository linked in the Datasets section of this publication.
 
 
 ## Experimental Procedure
 
 Our experimental procedure was designed to ensure a robust and comprehensive evaluation of the four imbalance handling scenarios across diverse datasets and models. The process consisted of the following steps:
 
 <h2> Dataset Splitting </h2>
 
 We employed a form of nested cross-validation for each dataset to ensure robust model evaluation and proper hyperparameter tuning:
 
 1. Outer Loop: 5-fold cross-validation
 
    - Each dataset was split into five folds
    - Results were reported for all five test splits, providing mean and standard deviation values across the folds
 
 2. Inner Validation: 90/10 train-validation split
    - For scenarios requiring hyperparameter tuning (SMOTE, Class Weights, and Decision Threshold Calibration), the training split from the outer loop was further divided into a 90% train and 10% validation split
    - The validation split was used exclusively for tuning hyperparameters
 
 This nested structure ensures that the test set from the outer loop remains completely unseen during both training and hyperparameter tuning, providing an unbiased estimate of model performance. The outer test set was reserved for final evaluation, while the inner validation set was used solely for hyperparameter optimization in the relevant scenarios.
 
 <h2> Scenario Descriptions </h2>
 We evaluated four distinct scenarios for handling class imbalance:
 
 1. **Baseline**: This scenario involves standard model training without any specific treatment for class imbalance. It serves as a control for comparing the effectiveness of the other strategies.
 
 2. **SMOTE (Synthetic Minority Over-sampling Technique)**: In this scenario, we apply SMOTE to the training data to generate synthetic examples of the minority class.
 
 3. **Class Weights**: This approach involves adjusting the importance of classes during model training, focusing on the minority class weight while keeping the majority class weight at 1.
 
 4. **Decision Threshold Calibration**: In this scenario, we adjust the classification threshold post-training to optimize the model's performance on imbalanced data.
 
 Each scenario implements only one treatment method in isolation. We do not combine treatments across scenarios. Specifically:
 
 - For scenarios 1, 2, and 3, we apply the default decision threshold of 0.5.
 - For scenarios 1, 2, and 4, the class weights are set to 1.0 for both positive and negative classes.
 - SMOTE is applied only in scenario 2, class weight adjustment only in scenario 3, and decision threshold calibration only in scenario 4.
 
 This approach allows us to assess the individual impact of each treatment method on handling class imbalance.
 
 
 <h2> Hyperparameter Tuning </h2>
 
 For scenarios requiring hyperparameter tuning (SMOTE, Class Weights, and Decision Threshold), we employed a simple grid search strategy to maximize the F1-score measured on the single validation split (10% of the training data) for each fold.
 
 The grid search details for the three treatment scenarios were as follows:
 
 <h3> SMOTE </h3>
 We tuned the number of neighbors hyperparameter, performing a simple grid search over `k` values of 1, 3, 5, 7, and 9.  
 <br/><br/>
 
 <h3> Class Weights </h3>
 In this scenario, we adjusted the class weights to handle class imbalance during model training. The tuning process involved adjusting the weight for the minority class relative to the majority class. If both classes were given equal weights (e.g., 1 and 1), no class imbalance handling was applied—this corresponds to the baseline scenario. For the balanced scenario, we set the minority class weight proportional to the class imbalance (e.g., if the majority/minority class ratio was 5:1, the weight for the minority class would be 5). We conducted grid search on the following factors: 0 (baseline case), 0.25, 0.5, 0.75, 1.0 (balanced), and 1.25 (over-correction). The optimal weight was selected based on the F1-score on the validation split.  
 <br/><br/>
 
 <h3> Decision Threshold Calibration </h3>
 We tuned the threshold parameter from 0.05 to 0.5 with a step size of 0.05, allowing for a wide range of potential decision boundaries.

:::info{title="Info"}
There are no scenario-specific hyperparameters to tune for the Baseline scenario. As a result, no train/validation split was needed, and the entire training set was used for model training.
:::

<h2> Overall Scope of Experiments </h2>
Overall, this study contains 9,000 experiments driven by the following factors:

- 30 datasets
- 15 models
- 4 scenarios
- 5-fold cross-validation

For each experiment, we recorded the 10 performance metrics across the five test splits. In the following sections, we present the results of these extensive experiments.

# Results

This section presents a comprehensive analysis of our experiments comparing four strategies for handling class imbalance in binary classification tasks. We begin with an overall comparison of the four scenarios (Baseline, SMOTE, Class Weights, and Decision Threshold Calibration) across all ten evaluation metrics. Following this, we focus on the F1-score metric to examine performance across the 15 classifier models and 30 datasets used in our study.

Our analysis is structured as follows:

1.  Overall performance comparison by scenario and metric
2.  Model-specific performance on F1-score
3.  Dataset-specific performance on F1-score
4.  Statistical analysis, including repeated measures tests and post-hoc pairwise comparisons

For the overall, model-specific, and dataset-specific analyses, we report mean performance and standard deviations across the five test splits from our cross-validation procedure. The final section presents the results of our statistical tests, offering a rigorous comparison of the four scenarios' effectiveness in handling class imbalance.

## Overall Comparison

Figure 1 presents the mean performance and standard deviation for all 10 evaluation metrics across the four scenarios: Baseline, SMOTE, Class Weights, and Decision Threshold Calibration.

![overall_results.svg](overall_results.svg)

_Figure 1: Mean performance and standard deviation of evaluation metrics across all scenarios. Best values per metric are highlighted in blue._

The results represent aggregated performance across all 15 models and 30 datasets, providing a comprehensive overview of the effectiveness of each scenario in handling class imbalance.

<h2> F1-Score Performance </h2>
 
 The results show that all three class imbalance handling techniques outperform the Baseline scenario in terms of F1-score:
 
 1. Decision Threshold Calibration achieved the highest mean F1-score (0.617 ± 0.005)
 2. SMOTE followed closely (0.605 ± 0.006)
 3. Class Weights showed improvement over Baseline (0.594 ± 0.006)
 4. Baseline had the lowest F1-score (0.556 ± 0.006)
 
 This suggests that addressing class imbalance, regardless of the method, generally improves model performance as measured by the F1-score.
 
 <h2> Other Metrics </h2>
 
 While our analysis primarily focuses on the F1-score, it's worth noting observations from the other metrics:
 
 - **F2-score and Recall**: Decision Threshold Calibration and SMOTE showed the highest performance, indicating these methods are particularly effective at improving the model's ability to identify the minority class.
 - **Precision**: The Baseline scenario achieved the highest precision, suggesting a more conservative approach in predicting the minority class.
 - **MCC (Matthews Correlation Coefficient)**: SMOTE and Decision Threshold Calibration tied for the best performance, indicating a good balance between true and false positives and negatives.
 - **PR-AUC and AUC**: These metrics showed relatively small differences across scenarios. Notably, SMOTE and Class Weights did not deteriorate performance on these metrics compared to the Baseline. As expected, Decision Threshold Calibration, being a post-model adjustment, does not materially impact these probability-based metrics (as well as Brier-Score).
 - **Accuracy**: The Baseline scenario achieved the highest accuracy, which is common in imbalanced datasets where high accuracy can be achieved despite poor minority class detection.
 - **Log-Loss**: The Baseline scenario performed best, suggesting it produces the most well-calibrated probabilities. SMOTE showed the highest log-loss, indicating potential issues with probability calibration.
 - **Brier-Score**: As expected, the Baseline and Decision Threshold scenarios show identical performance, as Decision Threshold Calibration is a post-prediction adjustment and doesn't affect the underlying probabilities used in the Brier Score calculation. Notably, SMOTE performed significantly worse on this metric, indicating it produces poorly calibrated probabilities compared to the other scenarios.
 
 Based on these observations, Decision Threshold Calibration demonstrates strong performance across several key metrics, particularly those focused on minority class prediction (F1-score, F2-score, and Recall). It achieves this without compromising the calibration of probabilities of the baseline model, as evidenced by the identical Brier Score. In contrast, while SMOTE improves minority class detection, it leads to the least well-calibrated probabilities, as shown by its poor Brier Score. This suggests that Decision Threshold Calibration could be particularly effective in scenarios where accurate identification of the minority class is crucial, while still maintaining the probability calibration of the original model.
 
 For the rest of this article, we will focus on the F1-score due to its balanced representation of precision and recall, which is particularly important in imbalanced classification tasks.

## Results by Model

Figure 2 presents the mean F1-scores and standard deviations for each of the 15 models across the four scenarios. Each model's scores are averaged across the 30 datasets.

![by_model_f1_results.svg](by_model_f1_results.svg)
_Figure 2: Mean F1-scores and standard deviations for each model across the four scenarios. Highest values per model are highlighted in blue._

Key observations from these results include:

1.  **Scenario Comparison**: For each model, we compared the performance of the four scenarios (Baseline, SMOTE, Class Weights, and Decision Threshold Calibration). This within-model comparison is more relevant than comparing different models to each other, given the diverse nature of the classifier techniques. </br>

2.  **Decision Threshold Performance**: The Decision Threshold Calibration scenario achieved the highest mean F1-score in 10 out of 15 models. Notably, even when it wasn't the top performer, it consistently remained very close to the best scenario for that model. </br>

3.  **Other Scenarios**: Within individual models, Class Weights performed best in 3 cases, while SMOTE and Baseline each led in 1 case. </br>

4.  **Consistent Improvement**: All three imbalance handling techniques generally showed improvement over the Baseline scenario across most models, with 1 exception. </br>

These results indicate Decision Threshold Calibration was most frequently the top performer across the 15 models. This suggests that post-model adjustments to the decision threshold is a robust strategy for improving model performance across different classifier techniques. However, the strong performance of other techniques in some cases underscores the importance of testing multiple approaches when dealing with imbalanced datasets in practice.

## Results by Dataset

Figure 3 presents the mean F1-scores and standard deviations for each of the 30 datasets across the four scenarios.

![by_dataset_f1_results.svg](by_dataset_f1_results.svg)

_Figure 3: Mean F1-scores and standard deviations for each dataset across the four scenarios. Highest values per dataset are highlighted in blue._

:::info{title="Info"}
These results are aggregated across the 15 models for each dataset. While this provides insights into overall trends, in practice, one would typically seek to identify the best model-scenario combination for a given dataset under consideration.
:::

Key observations from these results include:

1.  **Variability**: There is substantial variability in which scenario performs best across different datasets, highlighting that there is no one-size-fits-all solution for handling class imbalance.

2.  **Scenario Performance**:

    - Decision Threshold Calibration was best for 12 out of 30 datasets (40%)
    - SMOTE was best for 9 datasets (30%)
    - Class Weights was best for 7 datasets (23.3%)
    - Baseline was best for 3 datasets (10%)
    - There was one tie between SMOTE and Class Weights

3.  **Improvement Magnitude**: The degree of improvement over the Baseline varies greatly across datasets, from no improvement to substantial gains (e.g., satellite vs abalone_binarized).

4.  **Benefit of Imbalance Handling**: While no single technique consistently outperformed others across all datasets, the three imbalance handling strategies generally showed improvement over the Baseline for most datasets.

These results underscore the importance of testing multiple imbalance handling techniques for each specific dataset and task, rather than relying on a single approach. The variability observed suggests that the effectiveness of each method may depend on the unique characteristics of each dataset.

:::info{title="Info"}
One notable observation is the contrast between these dataset-level results and the earlier model-level results. While the model-level analysis suggested Decision Threshold Calibration as a generally robust approach, the dataset-level results show much more variability. This apparent discrepancy highlights the complexity of handling class imbalance and suggests that the effectiveness of different techniques may be more dependent on dataset characteristics than on model type.
:::

## Statistical Analysis

To rigorously compare the performance of the four scenarios, we conducted statistical tests on the F1-scores aggregated by dataset (averaging across the 15 models for each dataset).

 <h2> Repeated Measures ANOVA </h2>
 
 We performed a repeated measures ANOVA to test for significant differences among the four scenarios. For this test, we have 30 datasets, each with four scenario F1-scores, resulting in 120 data points. The null hypothesis is that there are no significant differences among the mean F1-scores of the four scenarios. We use Repeated Measures ANOVA to account because we have multiple measurements (scenarios) for each dataset.
 
 - **Result**: The test yielded a p-value of 2.01e-07, which is well below our alpha level of 0.05.
 - **Interpretation**: This result indicates statistically significant differences among the mean F1-scores of the four scenarios.
 
 <h2> Post-hoc Pairwise Comparisons </h2>
 
 Following the significant ANOVA result, we conducted post-hoc pairwise comparisons using a Bonferroni correction to adjust for multiple comparisons. With 6 comparisons, our adjusted alpha level is 0.05/6 = 0.0083.
 
 The p-values for the pairwise comparisons are presented in Table 1.
 
 **Table 1: P-values for pairwise comparisons (Bonferroni-corrected)**
 
 | Scenario           | Class Weights | Decision Threshold | SMOTE    |
 | ------------------ | ------------- | ------------------ | -------- |
 | Baseline           | 7.77e-05      | 2.26e-04           | 1.70e-03 |
 | Class Weights      | -             | 2.06e-03           | 1.29e-01 |
 | Decision Threshold | -             | -                  | 2.83e-02 |
 
 Key findings from the pairwise comparisons:
 
 1. The Baseline scenario is significantly different from all other scenarios (p < 0.0083 for all comparisons).
 2. Class Weights is significantly different from Baseline and Decision Threshold, but not from SMOTE.
 3. There is no significant difference between SMOTE and Decision Threshold, or between SMOTE and Class Weights at the adjusted alpha level.
 
 These results suggest that while all three imbalance handling techniques (SMOTE, Class Weights, and Decision Threshold) significantly improve upon the Baseline, the differences among these techniques are less pronounced. The Decision Threshold approach shows a significant improvement over Baseline and Class Weights, but not over SMOTE, indicating that both Decision Threshold and SMOTE may be equally effective strategies for handling class imbalance in many cases.

# Discussion of Results

 <h2> Key Findings and Implications </h2>
 
 Our comprehensive study on handling class imbalance in binary classification tasks yielded several important insights:
 
 1. **Addressing Class Imbalance**: Our results strongly suggest that handling class imbalance is crucial for improving model performance. Across most datasets and models, at least one of the imbalance handling techniques outperformed the baseline scenario, often by a significant margin. <br/><br/>
 
 2. **Effectiveness of SMOTE**: SMOTE demonstrated considerable effectiveness in minority class detection, showing significant improvements over the baseline in many cases. It was the best-performing method for 30% of the datasets, indicating its value as a class imbalance handling technique. However, it's important to note that while SMOTE improved minority class detection, it also showed the worst performance in terms of probability calibration, as evidenced by its high Log-Loss and Brier Score. This suggests that while SMOTE can be effective for improving classification performance, it may lead to less reliable probability estimates. Therefore, its use should be carefully considered in applications where well-calibrated probabilities are crucial. <br/><br/>
 
 3. **Optimal Method**: Decision Threshold Calibration emerged as the most consistently effective technique, performing best for 40% of datasets and showing robust performance across different model types. It's also worth noting that among the three methods studied, Decision Threshold Calibration is the least computationally expensive. Given its robust performance and efficiency, it could be considered a strong default choice for practitioners dealing with imbalanced datasets. <br/><br/>
 
 4. **Variability Across Datasets**: Despite the overall strong performance of Decision Threshold Calibration, we observed substantial variability in the best-performing method across datasets. This underscores the importance of testing multiple approaches for each specific problem. <br/><br/>
 
 5. **Importance of Dataset-Level Analysis**: Unlike many comparative studies on class imbalance that report results at the model level aggregated across datasets, our study emphasizes the importance of dataset-level analysis. We found that the best method can vary significantly depending on the dataset characteristics. This observation highlights the necessity of analyzing and reporting findings at the dataset level to provide a more nuanced and practical understanding of imbalance handling techniques.

 <h2> Study Limitations and Future Work </h2>
 
 While our study provides valuable insights, it's important to acknowledge its limitations:
 
 1. **Fixed Hyperparameters**: We used previously determined model hyperparameters. Future work could explore the impact of optimizing these hyperparameters specifically for imbalanced datasets. For instance, adjusting the maximum depth in tree models might allow for better modeling of rare classes. <br/><br/>
 
 2. **Statistical Analysis**: Our analysis relied on repeated measures ANOVA and post-hoc tests. A more sophisticated approach, such as a mixed-effects model accounting for both dataset and model variability simultaneously, could provide additional insights and is an area for future research. <br/><br/>
 
 3. **Dataset Characteristics**: While we observed variability in performance across datasets, we didn't deeply analyze how specific dataset characteristics (e.g., sample size, number of features, degree of imbalance) might influence the effectiveness of different methods. Future work could focus on identifying patterns in dataset characteristics that predict which imbalance handling technique is likely to perform best. <br/><br/>
 
 4. **Limited Scope of Techniques**: Our study focused on three common techniques for handling imbalance. Future research could expand this to include other methods or combinations of methods. <br/><br/>
 
 5. **Performance Metric Focus**: While we reported multiple metrics, our analysis primarily focused on F1-score. Different applications might prioritize other metrics, and the relative performance of these techniques could vary depending on the chosen metric.
 
 These limitations provide opportunities for future research to further refine our understanding of handling class imbalance in binary classification tasks. Despite these limitations, our study offers valuable guidance for practitioners and researchers dealing with imbalanced datasets, emphasizing the importance of addressing class imbalance and providing insights into the relative strengths of different approaches.

# Conclusion

Our study provides a comprehensive evaluation of three widely used strategies—SMOTE, Class Weights, and Decision Threshold Calibration—for handling imbalanced datasets in binary classification tasks. Compared to a baseline scenario where no intervention was applied, all three methods demonstrated substantial improvements in key metrics related to minority class detection, particularly the F1-score, across a wide range of datasets and machine learning models.

The results show that addressing class imbalance is crucial for improving model performance. Decision Threshold Calibration emerged as the most consistent and effective technique, offering significant performance gains across various datasets and models. SMOTE also performed well, and Class Weights tuning proved to be a reasonable method for handling class imbalance, showing moderate improvements over the baseline.

However, the variability in performance across datasets highlights that no single method is universally superior. Therefore, practitioners should consider testing multiple approaches and tuning them based on their specific dataset characteristics.

While our study offers valuable insights, certain areas could be explored in future research. We fixed the hyperparameters across scenarios to ensure fair comparisons, holding all factors constant except for the treatment. Future research could investigate optimizing hyperparameters specifically for imbalanced datasets. Additionally, further work could explore how specific dataset characteristics influence the effectiveness of different techniques. Expanding the scope to include other imbalance handling methods or combinations of methods would also provide deeper insights. While our primary analysis focused on the F1-score, results for other metrics are available, allowing for further exploration and custom analyses based on different performance criteria.

In conclusion, our findings emphasize the importance of addressing class imbalance and offer guidance on choosing appropriate techniques based on dataset and model characteristics. Decision Threshold Calibration, with its strong and consistent performance, can serve as a valuable starting point for practitioners dealing with imbalanced datasets, but flexibility and experimentation remain key to achieving the best results.

# Additional Resources

To support the reproducibility of our study and provide further value to researchers and practitioners, we have made several resources publicly available:

1.  **Model Repositories**: Implementations of all 15 models used in this study are available in separate repositories. These can be accessed through links provided in the "Models" section of this publication.<br><br>

2.  **Dataset Repository**: The 30 datasets used in our study are available in a GitHub repository titled "30 Imbalanced Classification Study Datasets". This repository includes detailed information about each dataset's characteristics and sources.

    - GitHub link: [https://github.com/readytensor/rt-datasets-binary-class-imbalance](https://github.com/readytensor/rt-datasets-binary-class-imbalance)

3.  **Results Repository**: A comprehensive collection of our study results is available in a GitHub repository titled "Imbalanced Classification Results Analysis". This includes detailed performance metrics and analysis scripts.

    - GitHub link: [https://github.com/readytensor/rt-binary-class-imbalance-results](https://github.com/readytensor/rt-binary-class-imbalance-results)

4.  **Hyperparameters**: The hyperparameters used in the experiment are listed in the **`hyperparmeters.csv`** file in the "Resources" section.

All project work is open-source, encouraging further exploration and extension of our research. We welcome inquiries and feedback from the community. For any questions or discussions related to this study, please contact the authors at contact@readytensor.com.

We encourage researchers and practitioners to utilize these resources to validate our findings, conduct further analyses, or extend this work in new directions.

# References

1.  Batista, G.E., Prati, R.C., & Monard, M.C. (2004). A study of the behavior of several methods for balancing machine learning training data. _ACM SIGKDD Explorations Newsletter_, 6(1), 20-29.
2.  Blagus, R., & Lusa, L. (2013). SMOTE for high-dimensional class-imbalanced data. _BMC Bioinformatics_, 14(1), 106.
3.  Bunkhumpornpat, C., Sinapiromsaran, K., & Lursinsap, C. (2009). Safe-level-SMOTE: Safe-level-synthetic minority over-sampling technique for handling the class imbalanced problem. In _Advances in Knowledge Discovery and Data Mining: 13th Pacific-Asia Conference, PAKDD 2009 Bangkok, Thailand, April 27-30, 2009 Proceedings_ (pp. 475-482). Springer Berlin Heidelberg.
4.  Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. (2002). SMOTE: Synthetic minority over-sampling technique. _Journal of Artificial Intelligence Research_, 16, 321-357.
5.  Elor, Y., & Averbuch-Elor, H. (2022). To SMOTE, or not to SMOTE? _arXiv preprint arXiv:2201.08528_.
6.  Han, H., Wang, W.Y., & Mao, B.H. (2005, August). Borderline-SMOTE: A new over-sampling method in imbalanced data sets learning. In _International Conference on Intelligent Computing_ (pp. 878-887). Berlin, Heidelberg: Springer Berlin Heidelberg.
7.  Kovács, G. (2019). SMOTE-variants: A Python implementation of 85 minority oversampling techniques. _Neurocomputing_, 366, 352-354.
8.  Van Hulse, J., Khoshgoftaar, T.M., & Napolitano, A. (2007, June). Experimental perspectives on learning from imbalanced data. In _Proceedings of the 24th International Conference on Machine Learning_ (pp. 935-942).
