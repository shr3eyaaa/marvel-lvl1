## Task 1: MATLAB ML Onramp Course

I completed the MATLAB Machine Learning Onramp course to learn the basics of practical machine learning. It consisted of 8 modules:

### Course Breakdown

1. **Overview of ML:** I learned the basics of practical machine learning and the ML workflow.
2. **Import Data:** I used functions like `readtable` to load a handwriting dataset, and used `plot` and `axis` to visualize the data.
3. **Extract Features:** I learned how features help distinguish between classes, and how to extract features and find their range.
4. **Partitioning Data:** I learned the difference between training and test data. I selected a fine KNN model and trained all the models.
5. **Train Models:** I selected a bunch of quick-to-train models and trained them. Then I selected a weighted KNN model and changed the model hyperparameters.
6. **Evaluate Performance:** I compared different models by training them and analyzing the confusion matrix - a table used to evaluate classification models by comparing real vs. predicted values.
7. **Improve Performance:** I applied feature selection to prevent low accuracy and overfitting.
8. **Conclusion:** Real-world applications and completed a wrap-up survey.

![Certificate](certificate_page-0001.jpg)

## Task 2: Kaggle Crafter - Build & Publish Your Own Dataset

I gathered data from public sources and compiled my own dataset on Kaggle. I watched YouTube videos to learn the fundamentals of data curation, documentation, and publication on the Kaggle website.

### The Dataset

I built a dataset in CSV format titled *"My Fav Dystopian and Sci-Fi Movies."* It contained the director, lead actor, franchise, runtime, IMDb rating, and more.

### Meeting the Usability Criteria

- **Completeness:** Added a subtitle, relevant tags, a full description, and a cover image.
- **Credibility:** Clearly documented the data's provenance and stated the update frequency.
- **Compatibility:** Published in CSV format with clear column descriptions, under a CC BY-SA 4.0 license.

### Result

The dataset achieved a Usability Score of 9.41, clearing the required threshold of ≥8.5.


[My dataset](https://www.kaggle.com/datasets/shr3eyaaa/my-fav-dystopian-and-sci-fi-movies-dataset)

## Task 5: Logistic Regression from Scratch

### Implementation
I built logistic regression from scratch (sigmoid + gradient descent, coded up sigmoid/loss/gradients/train/predict functions) and compared it to sklearn's built in version on the Framingham heart disease dataset.

### Results

| Model | Accuracy | Precision | Recall | F1 | Time |
|---|---|---|---|---|---|
| From Scratch | 0.859 | 0.714 | 0.134 | 0.226 | 0.62s |
| sklearn | 0.861 | 0.727 | 0.143 | 0.239 | 0.008s |

### Takeaways
- I got virtually identical results from both implementations, which confirmed my scratch gradient descent math was actually right.
- I noticed sklearn ran much faster because of its underlying optimizations, but my scratch version hit the exact same accuracy but just a lil slower and more manual.
- I saw low recall (~0.14) across both models, but I realized this was caused by dataset imbalance rather than an issue with my custom implementation.

![Logistic reg](imageoflr.png)

[Click here for code](https://github.com/shr3eyaaa/log-regression/blob/main/logistic_regression_article_code.py)

## Task 7: Fairness Meets Functionality - ID3 Algorithm

I used the Utrecht Fairness Recruitment Dataset from Kaggle and coded my own machine learning model. I learnt the fundamentals of the ID3 algorithm, model evaluation and bias detection in AI through YT videos.

### The Model
I built an ID3 Decision Tree from scratch in Python to predict hiring decisions. The model analyzed candidate data including age, gender, education level and years of experience to form its decision logic.

### Meeting the Evaluation & Fairness Criteria
- **Performance:** I tested the model using standard metrics and got around 72% accuracy, 65% precision, and a 61% F1-score.
- **Feature Importance:** I saw that `ExperienceYears` drove most of the decisions. It seems logical for hiring on paper, but it subtly pulls in indirect bias.
- **Demographic Parity:** I checked gender rates and found them pretty even (18.0% female vs 16.7% male), but I got a massive gap with age—the 35+ group had a 27.0% selection rate, while the under-25 group barely made 2.4%.
- **Equal Opportunity:** I split the qualified applicants by age and noticed the tree consistently missed younger candidates, showing a true +ve rate of only 5.6% for <25 versus 20.0% for 35+.

### Outcome
I got a solid baseline for accuracy, but I saw the model fail on age fairness because work experience essentially acted as a proxy for age.

[Code for this task](https://github.com/shr3eyaaa/id3-algo/blob/main/fairnesstest.py)

![result](fairness-result.png)


## Task 8: KNN with Ablation Study

I built a K-Nearest Neighbors classifier on the Breast Cancer Wisconsin dataset to predict malignant vs. benign tumors, then ran a feature ablation study to figure out which features actually matter for the model's predictions.

### Preprocessing & Setup

- **Cleaning & Encoding:** I dropped the `id` and `Unnamed: 32` columns, then mapped the diagnosis labels to binary values (M = 1, B = 0).
- **Splitting & Scaling:** I split the data 80/20 with stratification to keep class balance, then scaled all features using `StandardScaler` since KNN depends heavily on distance metrics.
- **Model Setup:** I trained a `KNeighborsClassifier` with k=5 and kept it fixed across every test so comparisons stayed fair.

### Feature Ablation & Findings

I removed one feature at a time, retrained the model from scratch each time, and tracked the shift in F1-score compared to my baseline.

- **Most Important:** I found that `fractal_dimension_mean`, `concavity_worst`, `texture_mean`, `concavity_mean`, and `smoothness_mean` caused the biggest drops in F1-score when removed.
- **Redundant Features:** I noticed most `_worst` metrics (`radius_worst`, `area_worst`, `perimeter_worst`, `texture_worst`) made zero difference to the score when dropped.
- **Noise Reduction:** I actually saw my F1-score improve when I removed `symmetry_worst`, showing it was just adding unnecessary noise to the distance calculations.

### Outcome

I confirmed that `fractal_dimension_mean` and `concavity_worst` carried most of the predictive weight, while many `_worst` features just overlapped with their `_mean` and `_se` counterparts. Trimming those extra features kept the model lighter without hurting accuracy.

[My code and results](https://github.com/shr3eyaaa/knn-ablation-code)

---


*Thank YOU*