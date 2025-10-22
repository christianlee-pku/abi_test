### Technical Challenge: Senior MLE (Computer Vision)

This document outlines the solution for the take-home challenge, covering the **Part A recognition prototype** and the **Part B validation strategy**.

### Part A: Recognition Prototype Implementation

#### 1. Approach and Technology Stack

The prototype simulates an **edge deployment** environment by exclusively relying on local, pre-trained models and libraries, adhering to the "No cloud/internet services" constraint.

| Component | Technology | Justification |
| :--- | :--- | :--- |
| **Video I/O** | `OpenCV (cv2)` | Industry standard for reliable video frame extraction. |
| **Core AI** | `dlib` via `face_recognition` | Provides fast HOG-based face detection and pre-trained ResNet-based 128D face embedding models, suitable for local, offline inference. |
| **Mathematics** | `NumPy` | Used for efficient vector manipulation and calculation of the Euclidean distance. |

#### 2. Robustness and Production Decisions

The implementation incorporates several critical decisions to handle real-world challenges, particularly the persistent environment issues encountered with `dlib` bindings:

| Decision Point | Value/Rationale | Production Relevance |
| :--- | :--- | :--- |
| **Robustness Fix** | **Explicit `np.uint8` casting** | Guarantees that the image array data type strictly conforms to the required C++ binding signature, resolving persistent `TypeError`s in the `compute_face_descriptor` call. | **Stability/Maintainability** |
| **Distance Calculation** | **Pure NumPy `np.linalg.norm`** | Bypasses the potentially buggy `face_recognition.face_distance` wrapper, ensuring reliable Euclidean distance calculation, irrespective of dependency conflicts. | **Reliability/System Integrity** |
| **Frame Skipping (N)** | `FRAME_SKIP_N=5` | **Optimization.** Reduces CPU load by processing only 1/5th of the frames. This is a critical trade-off between **Inference Speed** (high) and **Recall** (lower). The system prioritizes staying responsive on the edge device. | **Performance/Resource Management** |
| **Progress Display** | **`sys.stdout.write`** | Provides real-time feedback (Current Frame / Total Frames) to the operator, which is essential for monitoring long video processing jobs. | **Monitoring/User Experience** |

---

### Part B: Validation without Ground Truth

The validation strategy must be **practical** and focus on indirect, proxy, and business metrics, given the constraints (privacy, elderly population, no labeled video data).

#### Validation Strategy: Three-Pillar Approach

| Pillar | Focus | Specific Methods | Justification |
| :--- | :--- | :--- | :--- |
| **1. Offline Performance Proxy (Initial Benchmarking)** | **Model Integrity & Threshold Setting** | **Public Dataset Testing (e.g., AgeDB):** Use public datasets with age variance to simulate the target demographic. Plot ROC/FPR vs. FNMR (1-Recall) curves to determine the **optimal threshold $\tau$** that balances False Alarms (FPR) vs. Missed Detections (FNMR). | Sets a reliable **decision boundary** based on quantitative data before production deployment. |
| **2. Production Behavior Monitoring (Functional Success)** | **Business Value & Stability** | **Business Metric (Unlocking Success Rate):** Monitor the percentage of successful "Memory Unlocks" (Abi starting a conversation) out of total recognition attempts. High success rate is the ultimate proxy for system accuracy. **Non-Functional Monitoring:** Log inference latency (P95), CPU/memory usage, and system uptime on the edge device to ensure reliability. | Directly validates the system's **business value** and ensures **non-functional requirements** (speed, stability) are met under real-world conditions. |
| **3. Human-in-the-Loop Feedback (Qualitative Insight)** | **Calibration & Error Identification** | **Complaint Rate Tracking:** Systematically track the number of complaints (e.g., "Robot didn't recognize me" or "Robot recognized the wrong person") reported by caregivers. This provides real-world FNMR/FPR rates. **Low-Confidence Sampling:** Log only the *embeddings* (anonymized) and metadata (time, light conditions) for borderline detections ($\tau \pm \epsilon$) for internal QA review, avoiding privacy leaks. | Provides **actionable, real-world feedback** to incrementally adjust $\tau$ and pinpoint environmental issues. |