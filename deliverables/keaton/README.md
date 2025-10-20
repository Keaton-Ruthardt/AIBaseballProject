# Keaton's Deliverables - Group Leader

## Deliverable 1: Initial Setup & Research
**Task**: Collect initial video samples & set up team GitHub

### Detailed Explanation
Establish the project infrastructure by collecting diverse video samples of sacrifice play situations from various sources (MLB games, college baseball, etc.) and setting up the team's GitHub repository with proper structure, branching strategy, and collaboration guidelines. This foundation is critical for team coordination and ensuring we have quality data to work with throughout the project.

---

## Deliverable 2: Core System Development
**Task**: Build our player detection system and begin the player tracking implementation across frames

### Detailed Explanation
Develop the player detection system using chosen object detection models (such as YOLO or Faster R-CNN), implementing algorithms to identify and track individual players (runners, outfielders, infielders) consistently across multiple video frames. This involves handling challenges like occlusions, camera movements, and varying lighting conditions to maintain accurate tracking throughout plays.

---

## Deliverable 3: Feature Extraction & Validation
**Task**: Extract the computer vision features we will need such as outfielder position and runner's positioning

### Detailed Explanation
Implement computer vision algorithms to extract critical features including outfielder GPS-style coordinates at key moments (ball contact, ball caught), runner positioning and speed, and other spatial relationships that influence runner advancement decisions. These features will feed directly into our prediction model and must be accurate and reliable.

---

## Deliverable 4: Pipeline Integration & Optimization
**Task**: Complete the end to end pipeline video → detection → features grabbed → prediction

### Detailed Explanation
Orchestrate the complete end-to-end pipeline integration, ensuring seamless data flow from raw video input → player/ball detection → feature extraction → model prediction → output results. Implement proper error handling and logging at each stage to create a robust, production-ready system that can process videos reliably.

---

## Deliverable 5: Final Documentation & Demo
**Task**: Produce demo video (record, edit finalize 2-3min showcase)

### Detailed Explanation
Produce a polished 2-3 minute demo video showcasing the system analyzing real baseball footage, highlighting key features like player tracking, feature extraction, and real-time prediction. Include professional editing with graphics, annotations, and narration explaining the system's capabilities and impact for coaches and analysts.

