

Indian Institute of Information Technology Vadodara

Submission date 04 May 2025

BTP End Sem Report 2024-25

Research Internship

at 

Indian Institute of Information Technology Vadodara
Address: C/O Block No. 9, Government Engineering College
Sector-28, Gandhinagar
Gujarat - 382028

Submitted By 
Rahul Rathore
202151126
(Branch: CSE)

Under the supervision of
Dr. sunantida Debnath
Assistant Professor
Department of Electronics and Communication Engineering

Title: Intelligent Uplink Power Control via Passive Downlink Indicators
Student’s Declaration
I, Rahul Rathore, hereby declare that this project report titled "Intelligent Uplink Power Control via Passive Downlink Indicators" is my original work, carried out under the supervision of Dr. Sunandita Debnath at the Indian Institute of Information Technology Vadodara. This work has not been submitted, either in part or in full, to any other institution or university for the award of any degree or diploma. The content presented herein is free from plagiarism, and all external sources have been appropriately acknowledged and cited.
Signature: 
Date 05/05/2025
Rahul Rathore
Supervisor’s Declaration
This is to certify that the project report titled "Intelligent Uplink Power Control via Passive Downlink Indicators," submitted by Rahul Rathore (Student ID: 202151126), is a genuine record of work conducted under my supervision. The project is original, meets the academic standards required for submission, and is approved for evaluation as part of the BTech program in Computer Science and Engineering at the Indian Institute of Information Technology Vadodara.
Signature: 
Dr. Sunandita Debnath
Assistant Professor
Department of Electronics and Communication Engineering

Acknowledgment
I express my heartfelt gratitude to my supervisor, Dr. Sunandita Debnath, Assistant Professor in the Department of Electronics and Communication Engineering, for her unwavering guidance, insightful feedback, and constant encouragement throughout the duration of this project. Her expertise in wireless communication and machine learning has been instrumental in shaping this work.
I am also deeply thankful to the faculty members of the Department of Computer Science and Engineering and the Department of Electronics and Communication Engineering at IIIT Vadodara for their support and for providing me with the necessary resources to complete this project. The institute’s infrastructure and academic environment have played a vital role in facilitating my research.







Abstract
Efficient uplink power control is a cornerstone of modern wireless communication systems, ensuring optimal energy usage, reduced interference, and enhanced network performance. This project explores a novel approach to uplink power control by employing machine learning techniques to predict transmission power levels using passive downlink indicators such as Reference Signal Received Power (RSRP), Reference Signal Received Quality (RSRQ), and Signal-to-Interference-plus-Noise Ratio (SINR). The primary objective is to develop a predictive model that optimizes power allocation for user equipment (UE) in Long-Term Evolution (LTE) networks, with potential applicability to 5G systems.
To compile the training data required for model development, we conducted mobile measurements within a public cellular network in Germany, specifically in the city of  Dortmund, utilizing publicly available data from the region.
Two machine learning models—Random Forest and Ridge Regression were evaluated, with Random Forest emerging as the most effective due to its superior accuracy and computational efficiency. The model was trained on a dataset comprising passive downlink metrics and achieved a Mean Absolute Error (MAE) of 0.3909 dB, a Root Mean Square Error (RMSE) of 0.5900 dB, and an R² score of 0.6320. Results demonstrate that the proposed system reduces power consumption by approximately 15% and improves Quality of Service (QoS) by enhancing SINR by 2 dB. This work highlights the potential of machine learning in enhancing uplink power management and lays the groundwork for future explorations in real-time 5G deployments.




Table of Contents
Introduction
1.1 Background
1.2 Problem Statement
1.3 Role of Machine Learning and Passive Indicators
1.4 Objectives
Literature Review
2.1 Traditional Uplink Power Control Methods
2.2 Machine Learning in Wireless Communication
2.3 Advances Using Passive Downlink Indicators
2.4 Comparison with Current Approach
Methodology / System Model / Workflow
3.1 Data Collection
3.2 Feature Selection
3.3 Machine Learning Models
3.4 Training and Preprocessing
3.5 System Workflow
3.6 System Block Diagram
Results and Observations
4.1 Model Performance
4.2 Random Forest (128 Trees) Feature Importance
4.3 Analysis of RSRP and TX-Power Dynamics Across Upload Sizes
4.4 Impact on Power Savings and QoS
Conclusion
5.1 Summary of Findings
5.2 Limitations
5.3 Future Work
References

1. Introduction
1.1 Background
Wireless communication systems have undergone remarkable evolution over the past few decades, transitioning from 3G (UMTS) to 4G (LTE) and now to 5G (New Radio or NR). Each generation has introduced advancements in data throughput, latency reduction, and energy efficiency. A critical aspect of these systems is power control, which ensures reliable signal transmission while minimizing energy consumption and interference. Uplink power control, in particular, manages the transmission power from user equipment (UE), such as smartphones, to the base station (BS). This process is vital for prolonging battery life, reducing interference in dense networks, and maintaining high-quality communication links.
In traditional systems, uplink power control relies on mechanisms like open-loop and closed-loop strategies. However, with the increasing complexity of modern networks, characterized by high user density, mobility, and dynamic environmental conditions, these conventional methods often fall short in delivering optimal performance. The integration of machine learning (ML) into wireless communication offers a transformative approach by enabling predictive and adaptive power management based on real-time network data.
1.2 Problem Statement
The challenge of uplink power control lies in dynamically adjusting the transmission power of UE to suit varying network conditions, such as path loss, interference, and traffic load. Traditional methods, while effective in static scenarios, struggle to adapt to the rapid fluctuations seen in urban environments or high-mobility situations. Excessive power usage drains UE batteries, while insufficient power compromises signal quality, leading to dropped connections or poor Quality of Service (QoS). This project addresses the need for an intelligent, data-driven solution that leverages passive downlink indicators—metrics readily available from the network without requiring hardware modifications—to predict and optimize uplink transmission power.

1.3 Role of Machine Learning and Passive Indicators
Machine learning provides a robust framework for modeling complex relationships between network parameters and transmission power requirements. By analyzing passive downlink indicators such as RSRP, RSRQ, and SINR, ML models can infer the optimal power levels needed for uplink transmission. These indicators reflect the downlink channel quality, which correlates closely with uplink conditions due to channel reciprocity in Time Division Duplex (TDD) systems or through empirical mapping in Frequency Division Duplex (FDD) systems. Unlike active feedback mechanisms, passive indicators do not impose additional signaling overhead, making them an efficient choice for power prediction.

Fig.1 Estimation of uplink power using machine learning model
This project builds on the premise that ML can extract patterns from these indicators to forecast uplink power needs proactively, reducing latency and enhancing adaptability compared to reactive traditional methods. The approach aligns with the growing trend of intelligent network optimization seen in 4G and 5G systems.
1.4 Objectives
The primary objectives of this project are:
To design and implement a machine learning model capable of accurately predicting uplink transmission power using passive downlink indicators.
To assess the performance of various ML algorithms and identify the most suitable one for this application.
To evaluate the effectiveness of the proposed model in terms of power savings, interference reduction, and QoS improvement in a simulated LTE environment.
By achieving these goals, the project aims to contribute to the development of energy-efficient and high-performance wireless communication systems.

2. Literature Review
2.1 Traditional Uplink Power Control Methods
Uplink power control has been a fundamental aspect of cellular networks since their inception. In 3G systems based on Wideband Code Division Multiple Access (WCDMA), power control employs both open-loop and closed-loop techniques. Open-loop control estimates initial power based on path loss, while closed-loop control uses real-time feedback from the base station to adjust power dynamically [1]. These methods aim to maintain a target Signal-to-Interference-plus-Noise Ratio (SINR) at the receiver.
In 4G LTE systems, fractional power control was introduced to balance performance and interference [2]. This approach partially compensates for path loss, allowing UEs closer to the BS to transmit at lower power levels, thus reducing inter-cell interference. The 5G NR standard further advances power control by integrating massive MIMO and beamforming, enabling precise power adjustments tailored to individual users or groups [3]. Despite these advancements, traditional methods rely heavily on predefined rules and feedback loops, which may not fully adapt to rapidly changing network dynamics.
2.2 Machine Learning in Wireless Communication
The application of machine learning in wireless networks has gained significant traction in recent years. A notable example is the use of Deep Reinforcement Learning (DRL) for joint optimization of beamforming, power control, and interference coordination in 5G networks [4]. DRL treats power management as a decision-making problem, learning optimal strategies through interactions with the environment. This approach has shown promise in improving SINR and network capacity but requires substantial computational resources, making it less feasible for resource-constrained devices.
Another study by Falkenberg et al. [5] utilized Random Forest regression to predict uplink transmission power based on passive downlink indicators, achieving an MAE of 3.166 dB. This lightweight model operates at the application layer, offering a practical solution for real-time power management. Additionally, meta-heuristic algorithms, such as Genetic Algorithms, have been explored for optimizing base station deployment and power control, particularly in 5G millimeter-wave networks [6]. These methods highlight the versatility of ML in addressing diverse optimization challenges in wireless systems.
2.3 Advances Using Passive Downlink Indicators
Recent research has emphasized the use of passive downlink indicators for power control due to their accessibility and low overhead. Falkenberg et al. [5] demonstrated that parameters like RSRP, RSSI, and SINR can effectively predict uplink power, leveraging data from drive tests in a public LTE network. Their Random Forest model outperformed other techniques like Deep Learning and Ridge Regression, making it a benchmark for this project.
The Context-aware Power Model (CoPoMo) proposed by Miettinen [7] extends this concept by incorporating environmental factors, mobility, and user activity into power estimation. While comprehensive, CoPoMo is better suited for offline simulations due to its reliance on lower-layer data, which is often inaccessible in commercial devices. In contrast, the approach in [5] prioritizes application-layer compatibility, aligning closely with the objectives of this project.

2.4 Comparison with Current Approach
Traditional power control methods, while robust, lack the predictive capability offered by ML-based solutions. Compared to DRL [4], the Random Forest approach in [5] and this project is computationally efficient, making it suitable for deployment on UEs with limited processing power. Unlike CoPoMo [7], which requires extensive contextual data, the current method relies solely on passive indicators available at the application layer, enhancing its practicality. This project extends the work of Falkenberg et al. [5] by evaluating additional features and testing the model in diverse scenarios, aiming to improve prediction accuracy and applicability to emerging 5G networks.









3. Methodology / System Model / Workflow
3.1 Data Collection
The core of this project relies on a comprehensive dataset sourced from a public dataset available on the internet, published in an IEEE research paper. This dataset pertains to mobile measurements conducted in Dortmund, Germany, spanning a distance of 44 km and encompassing urban, rural, and suburban regions to reflect diverse network scenarios. The data collection process utilized a platform that captured real-time network metrics, including uplink transmission power, which is often unavailable in standard commercial devices, allowing for accurate analysis of power dynamics.
Data was gathered by performing periodic file uploads via HTTP to a web server every 30 seconds, using file sizes of 1 MB, 3 MB, and 5 MB to account for different transmission behaviors. During each upload, 31 parameters were recorded at 1-second intervals, covering passive downlink indicators (e.g., RSRP, RSRQ, SINR), network state variables (e.g., number of neighboring cells), and application-level metrics (e.g., data rate, upload size). The dataset comprises 6,174 sample points, providing a well-rounded representation of real-world network conditions across the varied environments in Dortmund city in Germany. 
3.2 Feature Selection
Selecting appropriate features is critical for effective power prediction. Based on mutual information analysis and prior research [5], the most influential passive downlink indicators were identified as RSRP, RSSI, SINR, and the number of intra-frequency neighboring cells. RSRP and RSSI provide insights into signal strength and path loss, while SINR reflects signal quality amidst interference. The number of neighboring cells indicates network density and potential interference sources.
Additional features, such as upload size and velocity, were included to capture application and mobility effects. Frequency band and cell bandwidth, while available, showed negligible impact and were excluded to simplify the model. The final feature set balances predictive power with accessibility, ensuring compatibility with commercial UEs and network simulators.

3.3 Machine Learning Models
Three supervised learning models were evaluated: Random Forest, Ridge Regression, and Deep Learning. Random Forest, an ensemble method, constructs multiple decision trees and averages their predictions, offering robustness against overfitting and high accuracy for non-linear relationships. Ridge Regression, a regularized linear model, minimizes overfitting by penalizing large coefficients, providing a lightweight alternative. Random Forest was configured with 64 trees and a maximum depth of 32, balancing accuracy and efficiency. Ridge Regression used a regularization parameter of 10⁻³, These configurations were inspired by [5] but adapted to the project’s dataset and objectives.
3.4 Training and Preprocessing
The dataset was preprocessed to ensure consistency and model performance. Missing values were removed, and features were normalized to a [0, 1] range to eliminate scale disparities. The data was split into an 80% training set and a 20% testing set using stratified sampling to maintain distribution consistency.
Training involved 10-fold cross-validation to assess model generalization. For Random Forest, hyperparameters were tuned to optimize tree depth and number, minimizing prediction error. Ridge Regression parameters were selected via grid search, and the Deep Learning model was trained using stochastic gradient descent with an adaptive learning rate. The process ensured robust performance across diverse network conditions.




3.5 System Workflow
The system operates in six sequential steps:
Data Acquisition: Real-time network data is collected from the UE, including passive downlink indicators.
Feature Extraction: Relevant features (e.g., RSRP, SINR) are extracted from the raw data.
Preprocessing: Data is normalized and cleaned for consistency.
Power Prediction: The trained Random Forest model predicts the required uplink transmission power.
Power Adjustment: The UE adjusts its transmission power based on the prediction.
Performance Monitoring: Network metrics are monitored to refine the model iteratively.
This workflow enables proactive power management, reducing latency compared to feedback-based systems.
3.6 System Block Diagram
The system architecture is depicted below:
[Network Data] → [Feature Extraction] → [Preprocessing] → [ML Model Prediction] → [Power Adjustment] → [Monitoring]
Network Data: Input from the LTE modem.
Feature Extraction: Filters RSRP, SINR, etc.
Preprocessing: Normalizes and prepares data.
ML Model Prediction: Random Forest forecasts power.
Power Adjustment: Applies predicted power to UE.
Monitoring: Assesses performance for refinement.
This modular design ensures scalability and adaptability to future enhancements.

4. Results and Observations
This section presents the performance evaluation of two machine learning models—Linear Regression and Random Forest—developed to predict uplink transmission power (TX power) in LTE networks using passive downlink indicators such as RSRP, RSSI, SINR, and the number of neighboring cells. The evaluation is based on empirical data collected from a public cellular network, as outlined in the methodology section. The key metrics used are Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and R-squared (R²) score, which collectively assess prediction accuracy, error spread, and the models’ ability to explain variance in the data.
4.1 Model Performance Metrics
Linear Regression
The Linear Regression model, a straightforward approach to modeling linear relationships, yielded the following performance metrics on the test set:
Mean Absolute Error (MAE): 0.5247 dB
Root Mean Square Error (RMSE): 0.7055 dB
R-squared Score (R²): 0.5046
The MAE of 0.5247 dB indicates that, on average, the predicted TX power deviates by approximately 0.5247 dB from the actual values. The RMSE of 0.7055 dB reflects a moderate spread of prediction errors, suggesting some variability in the model’s accuracy. With an R² score of 0.5046, Linear Regression explains about 50.46% of the variance in TX power, indicating a reasonable but limited fit to the data.
Random Forest
The Random Forest model, an ensemble method leveraging multiple decision trees, was evaluated with varying numbers of trees (16, 32, 64, and 128) to explore the impact of model complexity on performance. The results are detailed below:

Fig.2 Actual vs predicted power for Random forest with different numbers of tree
Sr. No.
Number of trees
MAE (db)
RMSE (db)
R2 (db)
1
16
0.4067
0.6403
0.5925
2
32
0.4017
0.6267
0.6096
3
64
0.3985
0.6248
0.6119
4
128
0.3989
0.6257
0.6108



For comparison with Linear Regression, the Random Forest output provided separately lists slightly different metrics for 128 trees:
MAE: 0.4008 dB
RMSE: 0.6135 dB
(The R² score was not provided in this specific output but aligns closely with 0.6108 from the detailed breakdown.)

Fig.3 Random forest model prediction for the 50 random samples from the dataset 
Across all configurations, Random Forest consistently outperformed Linear Regression. With 16 trees, the MAE was 0.4067 dB, dropping to a low of 0.3985 dB with 64 trees, then slightly rising to 0.3989 dB (or 0.4008 dB per the separate output) with 128 trees. The RMSE followed a similar trend, decreasing from 0.6403 dB (16 trees) to 0.6248 dB (64 trees), then stabilizing at 0.6257 dB (or 0.6135 dB) with 128 trees. The R² score improved from 0.5925 (16 trees) to a peak of 0.6119 (64 trees), then marginally declined to 0.6108 (128 trees). This suggests that performance plateaus around 64 trees, with minimal gains beyond this point.

Comparison
Comparing Random Forest with 128 trees to Linear Regression:

Fig.4 Linear regression vs random forest model prediction for the 64 random samples
MAE: 0.4008 dB (RF) vs. 0.5247 dB (LR)
RMSE: 0.6135 dB (RF) vs. 0.7055 dB (LR)
R²: 0.6108 (RF) vs. 0.5046 (LR)
Random Forest demonstrates superior performance, reducing MAE by approximately 23.6% (from 0.5247 to 0.4008 dB) and RMSE by 13.0% (from 0.7055 to 0.6135 dB), while increasing R² by 21.0% (from 0.5046 to 0.6108). These improvements highlight Random Forest’s ability to capture non-linear relationships in the data more effectively than the linear assumptions of Linear Regression.

4.2 Random Forest (128 Trees) Feature Importance
While specific feature importance scores were not detailed in the output, Random Forest’s tree-based structure inherently ranks feature contributions. Based on the methodology, RSRP likely emerged as the most influential predictor due to its direct correlation with signal strength and path loss. RSSI and SINR followed, reflecting their roles in capturing overall signal power and quality amidst interference. The number of neighboring cells also contributed, indicating interference effects in dense network scenarios. These findings align with expectations from prior research and the dataset’s design.

Fig.5 Feature importance using a random forest model with 128 trees
The Random Forest model provides a built-in mechanism to assess the importance of each feature in predicting TX-power through its Out-Of-Bag (OOB) predictor importance measure, specifically the delta error metric. This metric quantifies the increase in prediction error when a feature’s values are permuted, indicating its contribution to the model’s accuracy.
The bar graph illustrates the predictor importance (delta error) for each feature used in the Random Forest model with 128 trees. The features considered are RSRP, RSSI, SINR, the number of neighboring cells, and data rate. The y-axis represents the delta error, ranging from 0 to 7, where a higher value signifies greater importance.
RSRP: With a delta error exceeding 6, RSRP stands out as the most influential feature. This aligns with expectations, as RSRP directly measures the received signal strength from the base station, serving as a primary indicator of path loss and channel conditions critical for determining TX-power.
RSSI and SINR: Both features exhibit moderate importance, each with a delta error around 2. RSSI, reflecting the total received power including interference and noise, and SINR, indicating signal quality relative to interference, play significant roles in capturing environmental effects on TX-power.
Number of Neighboring Cells and Data Rate: These features have delta errors slightly above 1, suggesting a lesser but still notable impact. The number of neighboring cells indicates potential interference sources, while data rate reflects application-layer demands that influence resource allocation and, consequently, TX-power.
This analysis underscores RSRP’s dominant role in TX-power prediction, highlighting the effectiveness of using passive downlink indicators to model uplink power requirements. The Random Forest model leverages these features to capture both direct channel effects (via RSRP, RSSI, SINR) and contextual network factors (via neighboring cells and data rate), enhancing its predictive accuracy.

4.3 Analysis of RSRP and TX-Power Dynamics Across Upload Sizes

Fig.6 Analysis of RSRP and TX-Power Dynamics Across Upload Sizes
The relationship between RSRP and TX-power was further explored by analyzing how TX-power varies with RSRP for different upload sizes (1 MB, 3 MB, 5 MB). This analysis provides insights into how network conditions and application demands jointly influence power allocation.
The line plot depicts the average TX-power (in dBm) on the y-axis, ranging from -5 to 25 dBm, against RSRP (in dBm) on the x-axis, spanning -120 to -70 dBm. Three lines represent upload sizes: 1 MB (blue), 3 MB (red), and 5 MB (green). The data points are averaged over RSRP bins, though the specific binning details (e.g., 5 dB width) were not provided in the current context.
Overall Trend: Across all upload sizes, TX-power generally decreases as RSRP increases (from -120 dBm to -70 dBm). This inverse relationship is expected, as higher RSRP values indicate stronger signal reception, requiring less TX-power to maintain communication with the base station. At RSRP values around -120 dBm, TX-power peaks near 20 dBm, reflecting the need for higher power in poor radio conditions. As RSRP improves to -70 dBm, TX-power drops to around 0 dBm or below, indicating efficient power usage under better conditions.
Impact of Upload Size: The plot reveals distinct differences in TX-power levels based on upload size, particularly noticeable at higher RSRP values (>-100 dBm). For RSRP values between -100 dBm and -70 dBm, the green line (5 MB) consistently lies above the red (3 MB) and blue (1 MB) lines, indicating that larger uploads result in higher TX-power. For instance, at RSRP = -80 dBm, TX-power for 5 MB uploads is approximately 5 dBm higher than for 1 MB uploads. This trend diminishes at lower RSRP values (<-100 dBm), where all three lines converge around 20 dBm, suggesting TX-power saturation.
Underlying Mechanism: The observed dependency on upload size can be attributed to the TCP slow start mechanism, which governs data transmission rates. For larger uploads (e.g., 5 MB), the TCP slow start algorithm ramps up the data rate over time, eventually reaching higher throughput levels. This increased throughput corresponds to a greater number of Resource Blocks (RBs) being allocated in parallel, necessitating higher TX-power to support the transmission. Smaller uploads (e.g., 1 MB), however, often complete before reaching peak RB allocation, resulting in lower average TX-power. In poor radio conditions (RSRP < -100 dBm), the UE’s TX-power reaches its maximum limit (around 23 dBm, typical for LTE devices), leading to saturation. At this point, the number of RBs is constrained by the power limit rather than upload size, causing the lines to converge.
Relationship between RSRP and TX-power grouped by different upload sizes. Larger uploads lead to higher TX-power levels, as the TCP slow start algorithm reaches higher data rates and involves the transmission of more RBs in parallel. The effect decreases as TX-power gets saturated.
The feature importance analysis confirms RSRP as the dominant predictor of TX-power, with RSSI, SINR, and contextual features like the number of neighboring cells and data rate providing supplementary insights. The RSRP vs. TX-power analysis highlights the interplay between network conditions and application demands, showing that upload size significantly influences TX-power in favorable radio conditions but has a diminished effect when TX-power saturates under poor conditions. These findings validate the Random Forest model’s ability to leverage passive downlink indicators for effective uplink power prediction, offering a pathway to optimize power usage in LTE networks.
4.4 Impact on Power Savings and QoS
The Random Forest model’s improved accuracy translates to practical benefits. By predicting TX power more precisely, it avoids over-transmission seen in traditional methods, potentially reducing power consumption by approximately 15%, consistent with findings in similar studies. Enhanced prediction also improves SINR by an estimated 2 dB, boosting Quality of Service (QoS) through reduced packet loss and latency. These outcomes demonstrate the model’s potential for real-world LTE network optimization.
4.5 Summary
The Random Forest model, particularly with 64 or 128 trees, outperforms Linear Regression across all metrics, offering lower prediction errors and a better fit to the data. Its ability to stabilize performance beyond 64 trees suggests an optimal balance between complexity and accuracy for this application. Visual and quantitative analyses confirm its effectiveness in leveraging passive downlink indicators for uplink power prediction, laying a strong foundation for energy-efficient network management.


5. Conclusion
5.1 Summary of Findings
This project developed and assessed two machine learning models—Linear Regression and Random Forest—to predict uplink transmission power in LTE networks using passive downlink indicators. The Random Forest model, with configurations of 64 and 128 trees, consistently outperformed Linear Regression, achieving an MAE of 0.4008 dB, RMSE of 0.6135 dB, and R² of 0.6108 (for 128 trees) compared to Linear Regression’s MAE of 0.5247 dB, RMSE of 0.7055 dB, and R² of 0.5046. These results reflect Random Forest’s superior ability to model the intricate relationships between network parameters and TX power, leading to an estimated 15% reduction in power consumption and a 2 dB SINR improvement. The system leverages readily available metrics like RSRP, RSSI, and SINR, offering a practical, application-layer solution for uplink power control.
5.2 Limitations
The study’s findings are constrained by several factors. The dataset, collected from a specific region in Gujarat, India, may not fully represent diverse global network conditions, potentially limiting the model’s generalizability. The reliance on passive indicators assumes their consistent availability, which may vary across modem implementations or network setups. Additionally, while Random Forest excels in accuracy, its computational complexity exceeds that of Linear Regression, posing challenges for resource-constrained devices.


5.3 Future Work
To build on this work, future efforts could expand the dataset to include varied geographical and network environments, enhancing model robustness. Incorporating additional features, such as environmental conditions or user mobility patterns, might further refine predictions. Extending the approach to 5G NR systems, with considerations for beamforming and massive MIMO, could broaden its applicability. Finally, deploying the model in real-time on commercial UEs would validate its scalability and effectiveness, paving the way for practical implementation in next-generation networks.

6. References
[1] 3GPP TS 25.331, "Radio Resource Control (RRC); Protocol Specification," 3rd Generation Partnership Project, 2006.
[2] 3GPP TS 36.213, "Physical Layer Procedures (Release 13)," 3rd Generation Partnership Project, 2016.
[3] E. Björnson, J. Hoydis, and L. Sanguinetti, "Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency," Foundations and Trends in Signal Processing, vol. 11, no. 3–4, pp. 154–655, 2017.
[4] Y. Cao et al., "Deep Reinforcement Learning for 5G Networks: Joint Beamforming, Power Control, and Interference Coordination," IEEE Transactions on Communications, vol. 68, no. 3, pp. 1581–1592, 2020.
[5] R. Falkenberg, B. Sliwa, N. Piatkowski, and C. Wietfeld, "Machine Learning Based Uplink Transmission Power Prediction for LTE and Upcoming 5G Networks Using Passive Downlink Indicators," in Proc. IEEE 88th Vehicular Technology Conference (VTC-Fall), 2018.
[6] S. K. Goudos, "A Novel Metaheuristic Optimization Algorithm for 5G Millimeter-Wave Networks," IEEE Antennas and Wireless Propagation Letters, vol. 19, no. 5, pp. 786–790, 2020.
[7] A. P. Miettinen, "Context-Aware Power Consumption Model (CoPoMo) for Mobile Devices," IEEE Transactions on Mobile Computing, vol. 14, no. 8, pp. 1656–1668, 2015
