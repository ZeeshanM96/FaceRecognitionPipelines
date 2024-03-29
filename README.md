﻿# Face Recognition Pipelines

## Objective
The objective of this project is to conduct a detailed comparison between two leading face recognition frameworks, DeepFace and InsightFace, to assess their effectiveness and efficiency in processing and recognizing facial images. Our study aims to identify the strengths and weaknesses of each framework across various stages of face recognition, including detection, alignment, feature extraction, and matching, to provide insights into their applicability in real-world scenarios. Main idea was to come up with a plugNplay pipeline that all you need to do is integrate a model name or a function and thats it. Rest of the work will be done by the pipeline.

## Scope
The scope of our project includes the implementation of both DeepFace and InsightFace frameworks, the setup of a controlled testing environment using a standardized dataset (AgeDB), and the execution of a series of experiments designed to evaluate the performance of each framework under different conditions. This encompasses a comprehensive analysis of algorithmic effectiveness, computational efficiency, and scalability across diverse facial recognition tasks.

## Methodologies
The project implemented DeepFace and InsightFace frameworks for face recognition, using the AgeDB dataset for evaluation. The methodologies involved:

- Framework Implementation: Setting up both frameworks to process images from the AgeDB dataset.
- Performance Metrics: Accuracy, speed, and computational efficiency were the primary metrics for evaluation.
- Experimental Setup: Utilized Python and relevant libraries for implementation, with tests conducted on a specified hardware setup to ensure consistency.

## Results
- Accuracy: InsightFace demonstrated higher accuracy in recognizing faces compared to DeepFace, particularly in challenging conditions such as varied lighting and angles.
- Speed: InsightFace was faster in processing images, attributed to its efficient algorithmic optimizations.
- Computational Efficiency: DeepFace showed better performance in scenarios with limited computational resources, highlighting its suitability for applications with hardware constraints.

## Models used for the Pipeline of Insight Face:

- Face Detection:
   -- SCRFD (dte_10g, scrfd_10g_bnkps, scrfd_person_2.5g)
- Face landmarks detection:
   -- Simple Regression : 2d106det
- Face alignment:
   -- Retina face
- Feature extraction:
  -- Facenet ( by Deep Face)
  -- ArcFace
  -- VggFace
