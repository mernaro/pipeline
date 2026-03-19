# Unfolding: Pipeline

This project is part of our Master’s 2 program. It explores **super-resolution** and **image segmentation** here we implement a **Pipeline** with both parts.

**Basic pipeline** : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hIsxZMG-8RMj3xgpzD1IVrpgDaJlcxxF?usp=sharing)


**Dynamic Kernel Prior** : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xoavY70_M34iepVFoeJ8TJawwqVSbID9)

The pipeline using Dynamic Kernel Prior (DKP) is functional; however, the results from the segmentation stage are unsatisfying. We suspect that the RGB-to-grayscale conversion is the cause of this issue.
