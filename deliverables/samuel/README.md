# Samuel's Deliverables - Member 3

## Deliverable 1: Initial Setup & Research
**Task**: Analyze existing model and research feature extraction strategies

### Detailed Explanation
Perform an in-depth analysis of our existing runner advancement prediction model (which uses 45 features and ensemble methods including Random Forest, Gradient Boosting, and Logistic Regression). Understand its strengths, weaknesses, and which features are most critical (such as runner base position at 24.2% importance and hit distance at 20.3%). Set the foundation for integrating computer vision data with the current model architecture.

---

## Deliverable 2: Core System Development
**Task**: Create a simplified model reducing our features from 45 → 10

### Detailed Explanation
Work on model simplification, reducing the feature set from the current 45 features down to approximately 10 core features that can be reliably extracted from video alone (such as outfielder position, runner speed, ball hang time, hit distance, launch angle). Balance the trade-off between model accuracy and real-time prediction feasibility while maintaining performance close to the baseline 0.37 log loss.

---

## Deliverable 3: Feature Extraction & Validation
**Task**: Integrate model with feature extraction pipeline and run the initial predictions on test videos

### Detailed Explanation
Integrate the machine learning prediction model with the real-time feature extraction pipeline, enabling the system to process videos and generate runner advancement predictions. Conduct initial testing on video samples to identify integration issues, feature quality problems, and prediction accuracy. Debug any data format mismatches or pipeline failures.

---

## Deliverable 4: Pipeline Integration & Optimization
**Task**: Refine predictions and tune our confidence thresholds. Validate accuracy against ground truth data.

### Detailed Explanation
Fine-tune the prediction model for video-based features, adjusting confidence thresholds to optimize the precision/recall trade-off for practical use cases. Validate all predictions against ground truth data to ensure the system meets or exceeds the original model's performance (~0.37 log loss). Implement calibration techniques to ensure probability estimates are well-calibrated for decision-making.

---

## Deliverable 5: Final Documentation & Demo
**Task**: Complete technical report with all metrics and analysis

### Detailed Explanation
Author a comprehensive technical report documenting the complete methodology, feature engineering process, model architecture, experimental results with statistical analysis, performance metrics compared to baseline, limitations and failure cases, and recommendations for future improvements. This serves as the definitive technical reference for the project and demonstrates rigorous scientific approach to the problem.

