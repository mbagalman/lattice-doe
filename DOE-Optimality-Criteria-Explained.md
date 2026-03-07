# **The Architecture of Information: A Comprehensive Analysis of I, D, and A Optimality in Experimental Design**

The rigorous structured methodology known as Design of Experiments (DOE) represents a pivotal intersection between applied statistics and data-driven decision-making. For the early-career data scientist or analyst, DOE provides a formal framework for shifting from passive observational data collection to active, strategic information generation.1 Traditional data analysis often relies on historical datasets where variables may be highly correlated or inadequately sampled, leading to biased or incomplete models. In contrast, DOE empowers researchers to plan and conduct controlled tests that systematically manipulate input factors to evaluate their impact on a specified response or group of parameters.1 This transition is essential for rapid process optimization and identifying interactions that are frequently overlooked by the traditional "One-Factor-at-a-Time" (OFAT) approach.1  
Historically, experimental design was characterized by trial-and-error, but the modern statistical approach introduces sophisticated algorithms to design, conduct, and analyze experiments that multiply output efficiency while minimizing resource consumption.3 The fundamental objective of any DOE is to find the optimal factor levels that maximize an outcome more quickly and with fewer resources than traditional methods.4 This efficiency is achieved by manipulating multiple inputs simultaneously, allowing the researcher to identify not only the main effects of individual factors but also the synergized impacts of their interactions.1  
As experimental complexity increases—particularly in modern data science fields such as machine learning hyperparameter tuning or large-scale digital A/B/n testing—the application of "alphabetic optimality" becomes central to the design process.6 Optimality criteria serve as mathematical objective functions used by computer algorithms to select the most informative experimental runs from a vast candidate set of potential configurations.8 These criteria prioritize different statistical goals, such as maximizing the precision of model coefficients or minimizing the uncertainty of future predictions.6 By understanding the nuances of D, A, and I optimality, a practitioner can tailor their experimental strategy to the specific needs of their project, whether the goal is initial factor screening, precise effect estimation, or the development of a robust predictive surface.10

## **The Mathematical Engine of Optimal Design**

To appreciate the individual optimality criteria, it is necessary to examine the underlying linear model that forms the basis of experimental design. Most experiments are designed to fit a model of the form $Y \= X\\beta \+ \\epsilon$, where $Y$ is the vector of observed responses, $X$ is the model matrix (often called the design matrix), $\\beta$ represents the vector of unknown parameters or coefficients, and $\\epsilon$ denotes the random error associated with the measurements.4 The matrix $X$ is constructed based on the factor levels chosen for each experimental run and the mathematical form of the model the researcher intends to fit—be it a first-order linear model, a model with interactions, or a second-order quadratic model.8  
The quality of the parameter estimates $\\hat{\\beta}$ is governed by the variance-covariance matrix, which is defined as $\\text{Var}(\\hat{\\beta}) \= \\sigma^2(X^T X)^{-1}$, where $\\sigma^2$ represents the error variance of the process.13 The term $X^T X$ is referred to as the information matrix.2 A "good" design is one that, in some sense, maximizes the information matrix $X^T X$ or minimizes its inverse, $(X^T X)^{-1}$.14 Because minimizing a matrix is a multidimensional problem, statisticians have developed various scalar functions of the information matrix to serve as optimization targets; these functions are the alphabetic optimality criteria.5

| Criterion | Mathematical Target | Primary Statistical Focus | Typical Use Case |
| :---- | :---- | :---- | :---- |
| **D-Optimality** | $\\max | X^T X | $ |
| **A-Optimality** | $\\min \\text{Tr}((X^T X)^{-1})$ | Average variance of coefficients | Clinical trials, screening experiments |
| **I-Optimality** | $\\min \\int\_R x^T (X^T X)^{-1} x \\, d\\mu(x)$ | Average prediction variance over a space | Response surface optimization, prediction |
| **G-Optimality** | $\\min (\\max \\text{diag}(H))$ | Worst-case prediction variance | Protection against prediction outliers |
| **E-Optimality** | $\\max \\lambda\_{\\min}(X^T X)$ | Minimizing maximum variance of estimates | Robust parameter estimation |

The choice of criterion significantly influences the placement of experimental points in the design space. For example, some criteria tend to place points at the extreme boundaries of the design region to maximize the "leverage" on parameter estimation, while others distribute points more uniformly to ensure reliable prediction throughout the entire interior of the space.10

## **D-Optimality: Maximizing the Precision of Effect Estimates**

D-optimality is perhaps the most frequently applied criterion in the realm of computer-generated experimental designs.8 Its designation stems from its focus on the "Determinant" of the information matrix, making it the primary tool for researchers whose objective is to accurately estimate the coefficients of a statistical model.2

### **Intuition of D-Optimality**

The intuition behind D-optimality is best understood through the geometry of uncertainty. When a researcher estimates a set of model parameters, the joint uncertainty of those estimates can be visualized as a multidimensional confidence ellipsoid in the parameter space.6 The volume of this ellipsoid represents the total generalized variance of the estimates; a larger volume indicates that the parameters are estimated with less overall precision, while a smaller volume indicates higher precision.13  
D-optimality aims to minimize the volume of this confidence ellipsoid.6 Mathematically, the volume is inversely proportional to the square root of the determinant of the information matrix, $|X^T X|$.14 Therefore, by maximizing the determinant, D-optimality "shrinks" the joint uncertainty of all parameters simultaneously.6 This criterion is particularly effective for screening experiments, where the goal is to differentiate "signal" from "noise" and identify which factors have a statistically significant impact on the response.10 However, it is fundamentally model-dependent: if the researcher specifies a linear model but the true relationship is quadratic, the D-optimal design—which often places points only at the extreme corners—may fail to detect the curvature.5

### **Realistic-but-Simple Example: Industrial Process Screening**

Consider a chemical manufacturing process where a team of engineers wants to identify which factors influence the strength of a new adhesive bond.1 They suspect that three factors might be critical: Temperature, Pressure, and the Concentration of a specific curing agent.1 Due to the high cost of raw materials and limited laboratory time, the team can only afford 12 experimental runs.8  
A standard full factorial design at multiple levels might require 27 or more runs, which exceeds the budget.8 Furthermore, specific high-temperature/high-pressure combinations are unsafe and must be excluded from the design space.5 To solve this, the data scientist creates a "candidate set" of all feasible factor combinations and applies a D-optimal algorithm.8 The algorithm selects the 12 specific combinations that maximize the determinant of the information matrix for the suspected linear model.8 The resulting design provides the most precise estimates of the main effects of Temperature, Pressure, and Concentration given the constraints, allowing the team to confidently identify the active factors for further study.8

### **Technical Detail of D-Optimality**

Technically, a D-optimal design $d^\*$ is one that maximizes the determinant of the information matrix over the set of all possible designs $\\mathcal{D}$:

$$d^\* \= \\arg \\max\_{d \\in \\mathcal{D}} |X\_d^T X\_d|$$  
This is equivalent to minimizing the determinant of the variance-covariance matrix $|(X\_d^T X\_d)^{-1}|$.2 A key property of D-optimality is that it is invariant to non-singular recoding of the design matrix.14 This means the optimal design will remain the same regardless of whether the factors are measured in original units (e.g., Kelvin) or coded units (e.g., \-1 to \+1).14  
The quality of a design is often assessed using D-efficiency, calculated as:

$$D\_{\\text{eff}} \= 100 \\times \\left( \\frac{|X^T X|^{1/p}}{n} \\right)$$  
where $p$ is the number of parameters and $n$ is the number of runs.8 While standard orthogonal designs like fractional factorials often achieve 100% D-efficiency for simple models, computer-generated designs in constrained spaces typically yield lower scores, though they remain the most efficient choice available under those specific restrictions.8 One advanced variation is Bayesian D-optimality, which allows the practitioner to designate certain effects as "Necessary" to estimate and others as "If Possible," allowing for a design that estimates primary effects precisely while still providing some ability to detect higher-order interactions.10

## **A-Optimality: Minimizing Average Variance of Parameter Estimates**

While D-optimality focuses on the joint precision of all parameters, A-optimality (short for "Average") shifts the focus to the individual variances of the estimated coefficients.6 This criterion is particularly valued by practitioners who intend to perform separate hypothesis tests or construct individual confidence intervals for each factor in their model.13

### **Intuition of A-Optimality**

A-optimality seeks to minimize the average variance of the estimated regression coefficients.5 In practical terms, when a data scientist performs a t-test to determine if a specific factor is significant, the power of that test is directly linked to the standard error of that factor's coefficient.13 A-optimality concentrates on these standard errors directly.  
Unlike D-optimality, which might allow some parameters to have relatively higher variances as long as the overall "volume" of uncertainty is small, A-optimality forces the design to be more balanced in its precision.6 It is essentially the "budget-conscious" approach to precision, ensuring that no single parameter's estimation is neglected.6 Recent research has even argued that for factor screening—identifying an active subset of factors—A-optimal designs may be superior to D-optimal designs because they often result in less correlation between the columns of the model matrix, making the results easier to interpret and the tests more powerful.13

### **Realistic-but-Simple Example: Clinical Trial Design**

In the context of a clinical trial, a researcher might be comparing the efficacy of three different pharmaceutical formulations against a placebo.6 The primary goal is to estimate the treatment effect of each formulation with sufficient precision to decide which drug should advance to the next stage of regulatory approval.6  
If the researcher views each formulation as equally important, they would employ an A-optimal design.6 This design would distribute the available patients across the four groups (Placebo, Drug A, Drug B, Drug C) in a manner that minimizes the sum of the variances of the estimated treatment effects.6 In a simple, unconstrained case, this would likely lead to equal allocation, but if the costs of administering the different drugs vary, the A-optimal algorithm would adjust the allocation to find the most cost-effective way to achieve high average precision.13

### **Technical Detail of A-Optimality**

Mathematically, an A-optimal design minimizes the trace of the inverse of the information matrix:

$$d^\* \= \\arg \\min\_{d \\in \\mathcal{D}} \\text{Tr}((X\_d^T X\_d)^{-1})$$  
The trace of a matrix is the sum of its diagonal elements.2 Since the diagonal elements of the variance-covariance matrix $(X^T X)^{-1}$ represent the variances of the individual parameter estimates, minimizing the trace is identical to minimizing the sum (and thus the average) of those variances.5  
A critical technical limitation of A-optimality is that it is not invariant to the scaling or coding of the factors.13 If one factor is measured on a scale of 0 to 1 and another on a scale of 0 to 1,000, the A-optimal design will be heavily biased by the units of measurement.14 Consequently, it is mandatory to use orthogonally coded factors (standardizing all factors to a consistent range like \-1 to \+1) before applying the A-optimality criterion to ensure that the results are statistically meaningful and not artifacts of the chosen units.14

## **I-Optimality: Optimization for Prediction and Response Surfaces**

In many modern data science applications, the primary interest is not in the individual coefficients of the model, but in the model's ability to make accurate predictions about future observations.9 When the objective is to find the "optimal" settings for a process—a common goal in Response Surface Methodology (RSM)—I-optimality (also known as IV-optimality or Integrated optimality) is the preferred criterion.10

### **Intuition of I-Optimality**

I-optimality is fundamentally prediction-oriented.6 When a model is used to predict a response at a new point in the experimental space, there is a certain amount of uncertainty associated with that prediction, known as the prediction variance.11 This variance is typically not uniform across the space; it often fluctuates, usually being lower near the center of the experimental region and increasing as one moves toward the boundaries.11  
The "I" in I-optimality stands for "Integrated," reflecting the fact that the criterion aims to minimize the average prediction variance across the entire design space.5 While D-optimality might place many points at the extremes of the space to get a better estimate of the "slopes" (parameters), I-optimality often places more points in the interior of the space.10 This ensures that the model predicts well throughout the entire region, which is vital when the researcher's goal is to identify the precise combination of factor levels that maximizes or minimizes the response.11

### **Realistic-but-Simple Example: Predictive Model for Software Latency**

Suppose a data analyst is investigating how different server configurations (e.g., CPU allocation, Memory limits, and Disk I/O priority) affect the latency of a web application.18 The ultimate goal is to build a "digital twin" or a predictive surface that allows developers to estimate the latency for any arbitrary server configuration within a certain operating range.18  
Because the primary objective is to make precise predictions of latency across the entire range of configurations, the analyst chooses an I-optimal design.10 Unlike a screening design that might only test the "maximum" and "minimum" settings, the I-optimal design will distribute experimental runs throughout the configuration space.10 This ensures that the resulting latency surface is accurate everywhere, not just at the corners, allowing the development team to reliably optimize their infrastructure.10

### **Technical Detail of I-Optimality**

Technically, I-optimality minimizes the integral of the prediction variance over the region of interest $R$:

$$d^\* \= \\arg \\min\_{d \\in \\mathcal{D}} \\frac{1}{\\text{Vol}(R)} \\int\_R x^T (X\_d^T X\_d)^{-1} x \\, d\\mu(x)$$  
where $x$ is a vector representing a point in the design space and $x^T (X^T X)^{-1} x$ is the scaled prediction variance at that point.10 In practice, if the region $R$ consists of a discrete set of candidate points, I-optimality becomes equivalent to V-optimality, which minimizes the average prediction variance over those specific points.2  
I-optimality is often the default choice for second-order quadratic models where optimization of the response is the priority.10 An interesting mathematical property is that for orthogonally coded designs, I-optimality is equivalent to A-optimality for a transformed version of the model, allowing researchers to use efficient A-optimality search algorithms to generate I-optimal designs.14

## **Comparative Analysis of the Optimality Criteria**

Deciding which criterion to use requires a clear articulation of the experimental goal. The trade-offs between D, A, and I optimality are well-documented and represent the fundamental choices a researcher must make during the planning phase.6

| Feature | D-Optimality | A-Optimality | I-Optimality |
| :---- | :---- | :---- | :---- |
| **Statistical Goal** | Precise parameter estimation 10 | Precise individual effects 6 | Precise response prediction 10 |
| **Geometric Intuition** | Minimize ellipsoid volume 6 | Minimize average variance 6 | Minimize average prediction error 11 |
| **Common Application** | Factor screening 10 | Clinical trials, screening 6 | Response surface modeling 12 |
| **Model Dependency** | High 5 | High 5 | High 5 |
| **Scale Invariance** | Yes 14 | No (coding required) 14 | Yes 14 |
| **Point Placement** | Mostly at extremes 10 | Mixed 13 | Distributed throughout space 10 |

The General Equivalence Theorem provides a deep theoretical bridge between these criteria, stating that under certain ideal conditions, a design that is D-optimal is also G-optimal (minimizing the maximum prediction variance).6 However, in practical applications with finite runs and non-standard design spaces, the I-optimal design will almost always provide better average prediction than a D-optimal design, while the D-optimal design will provide better precision for estimating coefficients.11

## **Expanding the Optimality Spectrum**

While D, A, and I criteria are the pillars of optimal design, the "alphabet" extends to other specialized criteria developed to solve specific experimental challenges. Acknowledge these allows the data scientist to recognize when a standard approach might be insufficient.5

* **G-Optimality:** Focused on protecting against the "worst-case" scenario for prediction, G-optimality minimizes the maximum prediction variance in the design space.5 It is a conservative, minimax approach often used in high-stakes engineering where extreme errors must be avoided.6  
* **E-Optimality:** This criterion minimizes the maximum variance of any linear combination of the parameter estimates by maximizing the minimum eigenvalue of the information matrix.5 Geometrically, it seeks to minimize the length of the longest axis of the uncertainty ellipsoid.5  
* **V-Optimality:** Closely related to I-optimality, V-optimality minimizes the average prediction variance over a specific set of discrete points rather than an entire continuous region.2  
* **T-Optimality:** Used primarily for model discrimination, T-optimality designs experiments to maximize the lack-of-fit difference between two competing models, helping researchers decide which model best explains the data.5  
* **C-Optimality:** This criterion minimizes the variance of a specific linear combination of parameters, which is useful when the researcher cares about a particular weighted sum of effects rather than the individual coefficients.5

## **Modern Context: DOE in Machine Learning and Digital Experimentation**

The transition of DOE from traditional manufacturing into the digital world has opened new horizons for optimality criteria. In the field of machine learning, DOE is increasingly used for hyperparameter optimization.7 Training a complex deep neural network can require massive computational effort; rather than using a "Grid Search" which tests every combination, or a "Random Search," data scientists use D-optimal or orthogonal designs to pick a strategically small number of hyperparameter settings to test.7 This allows them to model the "accuracy surface" of the network and find the optimal settings with a fraction of the traditional computational cost.7  
Similarly, in digital product development, A/B/n testing has evolved into multivariate testing (MVT).27 When a marketing team wants to test dozens of variations across a landing page (e.g., different headlines, images, button colors, and layouts), they cannot show every combination to every user.27 By applying D-optimal designs, the team can test a mathematically chosen subset of these variations and still calculate the impact of each individual element and their interactions with high statistical confidence.27

## **Practical Implementation and Algorithmic Search**

Because finding a design that maximizes a determinant or minimizes a trace is a complex combinatorial problem, optimal designs are almost exclusively generated using computer software.5 Most modern statistical packages employ iterative exchange algorithms, such as the Fedorov algorithm or the coordinate exchange algorithm.5  
The generation process typically involves:

1. **Candidate Selection:** The user specifies the factors, their ranges, and the model to be fit. The software generates a "candidate set" of potential experimental points.8  
2. **Initial Design:** The algorithm picks a random subset of points to form a starting design.16  
3. **Stepwise Exchange:** The algorithm systematically swaps points in the current design with points from the candidate set, checking if the optimality criterion (e.g., the D-determinant) improves with each swap.8  
4. **Convergence:** This process continues until no further improvements are possible. To avoid getting stuck in a local optimum, the software usually repeats this process multiple times with different random starts.14

A persistent challenge for the data scientist is model robustness.5 Because an optimal design is tailored specifically to the model the researcher *thinks* is true, it can be inefficient if that model is wrong.5 For this reason, many practitioners include "center points" or use Bayesian designs to ensure that the experiment has some capability to detect "lack of fit" and hidden curvature in the data.10

## **Conclusion: Selecting the Right Criterion for the Objective**

For the early-career data scientist, the mastery of optimality criteria represents the difference between running an experiment and designing a strategy. The choice of criterion is a direct reflection of the experiment's value proposition. If the primary need is to explore a new system and identify which levers have the most impact, D-optimality provides a robust, scale-invariant default for factor screening.8 If the goal is to precisely measure and test individual effects—perhaps for regulatory compliance or scientific validation—A-optimality offers a balanced focus on parameter variances.6 Finally, when the objective is to build a high-performance predictive model or to optimize a response surface, I-optimality stands alone as the most effective tool for ensuring prediction quality across the entire region of interest.10  
By understanding these mathematical objectives and leveraging the power of modern computer-aided design software, analysts can ensure that every experimental run is maximized for information, minimizing the time and cost required to turn data into actionable insight.4 As datasets grow and experimentation becomes more expensive, the principles of optimal design will continue to be a cornerstone of professional data science practice.

#### **Works cited**

1. What Is Design of Experiments (DOE)? \- ASQ, accessed March 7, 2026, [https://asq.org/quality-resources/design-of-experiments](https://asq.org/quality-resources/design-of-experiments)  
2. Design Evaluation and Power Study, accessed March 7, 2026, [https://help.reliasoft.com/reference/experiment\_design\_and\_analysis/doe/design\_evaluation\_and\_power\_study.html](https://help.reliasoft.com/reference/experiment_design_and_analysis/doe/design_evaluation_and_power_study.html)  
3. Article: Step-by-Step Guide to DoE (Design of Experiments) \- SixSigma.us, accessed March 7, 2026, [https://www.6sigma.us/six-sigma-articles/step-by-step-guide-to-doe-design-of-experiments/](https://www.6sigma.us/six-sigma-articles/step-by-step-guide-to-doe-design-of-experiments/)  
4. A beginner's guide to design of experiments for life sciences using Spreadsheet DoE | The Biochemist | Portland Press, accessed March 7, 2026, [https://portlandpress.com/biochemist/article/46/6/29/235488/A-beginner-s-guide-to-design-of-experiments-for](https://portlandpress.com/biochemist/article/46/6/29/235488/A-beginner-s-guide-to-design-of-experiments-for)  
5. Optimal experimental design \- Wikipedia, accessed March 7, 2026, [https://en.wikipedia.org/wiki/Optimal\_experimental\_design](https://en.wikipedia.org/wiki/Optimal_experimental_design)  
6. 15.2 Alphabetic optimality criteria (A, D, E, G-optimality) \- Fiveable, accessed March 7, 2026, [https://fiveable.me/experimental-design/unit-15/alphabetic-optimality-criteria-a-d-e-g-optimality/study-guide/AvvUCe8w6phXKuyA](https://fiveable.me/experimental-design/unit-15/alphabetic-optimality-criteria-a-d-e-g-optimality/study-guide/AvvUCe8w6phXKuyA)  
7. Design of Experiment-based Configuration of Hyperparameters Of An Artificial Neural Network \- American Statistical Association, accessed March 7, 2026, [https://ww2.amstat.org/meetings/proceedings/2020/data/assets/pdf/1505419.pdf](https://ww2.amstat.org/meetings/proceedings/2020/data/assets/pdf/1505419.pdf)  
8. 5.5.2.1. D-Optimal designs \- Information Technology Laboratory, accessed March 7, 2026, [https://www.itl.nist.gov/div898/handbook/pri/section5/pri521.htm](https://www.itl.nist.gov/div898/handbook/pri/section5/pri521.htm)  
9. The Evolution of Experimental Design | by Victor Guiller \- Medium, accessed March 7, 2026, [https://medium.com/@victor.guiller/the-evolution-of-experimental-design-bf5b9b195476](https://medium.com/@victor.guiller/the-evolution-of-experimental-design-bf5b9b195476)  
10. Optimality Criteria \- JMP, accessed March 7, 2026, [https://www.jmp.com/support/help/en/19.0/jmp/optimality-criteria.shtml](https://www.jmp.com/support/help/en/19.0/jmp/optimality-criteria.shtml)  
11. I-optimal or G-optimal: Do We Have to Choose? \- DigitalCommons@USU, accessed March 7, 2026, [https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=1423\&context=mathsci\_facpub](https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=1423&context=mathsci_facpub)  
12. What are “optimal designs” and “optimality criteria”? \- JMP User Community, accessed March 7, 2026, [https://community.jmp.com/t5/JMPer-Cable/What-are-optimal-designs-and-optimality-criteria/ba-p/820437](https://community.jmp.com/t5/JMPer-Cable/What-are-optimal-designs-and-optimality-criteria/ba-p/820437)  
13. A-Optimal versus D-Optimal Design of Screening Experiments \- Lirias, accessed March 7, 2026, [https://lirias.kuleuven.be/retrieve/584944](https://lirias.kuleuven.be/retrieve/584944)  
14. Optimality Criteria, accessed March 7, 2026, [https://www.sfu.ca/sasdoc/sashtml/qc/chap24/sect30.htm](https://www.sfu.ca/sasdoc/sashtml/qc/chap24/sect30.htm)  
15. Statistics for Experimentalists Prof. Kannan. A Department of Chemical Engineering Indian Institute of Technology – Madras Lec, accessed March 7, 2026, [http://acl.digimat.in/nptel/courses/video/103106112/lec53.pdf](http://acl.digimat.in/nptel/courses/video/103106112/lec53.pdf)  
16. D-Optimal Designs \- NCSS, accessed March 7, 2026, [https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/PASS/D-Optimal\_Designs.pdf](https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/PASS/D-Optimal_Designs.pdf)  
17. A-optimal versus D-optimal design of screening experiments \- ASQ, accessed March 7, 2026, [https://asq.org/quality-resources/articles/a-optimal-versus-d-optimal-design-of-screening-experiments?id=d10f009a0cef4041908ee29676b4a871](https://asq.org/quality-resources/articles/a-optimal-versus-d-optimal-design-of-screening-experiments?id=d10f009a0cef4041908ee29676b4a871)  
18. Data Science Algorithms Explained with Examples \- Svitla Systems, accessed March 7, 2026, [https://svitla.com/blog/data-science-algorithms-explained-on-real-life-examples/](https://svitla.com/blog/data-science-algorithms-explained-on-real-life-examples/)  
19. Why Every Data Scientist Should Learn Mathematical Optimization | Towards Data Science, accessed March 7, 2026, [https://towardsdatascience.com/why-every-data-scientist-should-learn-mathematical-optimization-3ac102663456/](https://towardsdatascience.com/why-every-data-scientist-should-learn-mathematical-optimization-3ac102663456/)  
20. Optimization. The three pillars of Data Science are: | by Heena Rijhwani | Analytics Vidhya, accessed March 7, 2026, [https://medium.com/analytics-vidhya/optimization-acb996a4623c](https://medium.com/analytics-vidhya/optimization-acb996a4623c)  
21. When would you use an I-optimal design over a D-optimal design? \- Quora, accessed March 7, 2026, [https://www.quora.com/When-would-you-use-an-I-optimal-design-over-a-D-optimal-design](https://www.quora.com/When-would-you-use-an-I-optimal-design-over-a-D-optimal-design)  
22. When would you use an I-optimal design over a D-optimal design? \- Stats StackExchange, accessed March 7, 2026, [https://stats.stackexchange.com/questions/140526/when-would-you-use-an-i-optimal-design-over-a-d-optimal-design](https://stats.stackexchange.com/questions/140526/when-would-you-use-an-i-optimal-design-over-a-d-optimal-design)  
23. Optimal Design, accessed March 7, 2026, [https://statweb.rutgers.edu/buyske/591/lect11.pdf](https://statweb.rutgers.edu/buyske/591/lect11.pdf)  
24. Using design of experiments during the process of tuning hyperparameters in machine learning algorithms \- TU e-Thesis (Thammasat University), accessed March 7, 2026, [http://ethesisarchive.library.tu.ac.th/thesis/2021/TU\_2021\_6422040201\_15955\_19892.pdf](http://ethesisarchive.library.tu.ac.th/thesis/2021/TU_2021_6422040201_15955_19892.pdf)  
25. Evaluating Designs for Hyperparameter Tuning in Deep Neural Networks, accessed March 7, 2026, [https://nejsds.nestat.org/journal/NEJSDS/article/27](https://nejsds.nestat.org/journal/NEJSDS/article/27)  
26. Efficient Design of Machine Learning Hyperparameter Optimizers \- Imperial College London, accessed March 7, 2026, [https://www.imperial.ac.uk/media/imperial-college/faculty-of-engineering/computing/public/distinguished-projects/1819-ug-projects/MatacheC-Efficient-Design-of-Machine-Learning-Hyperparameter-Optimizers.pdf](https://www.imperial.ac.uk/media/imperial-college/faculty-of-engineering/computing/public/distinguished-projects/1819-ug-projects/MatacheC-Efficient-Design-of-Machine-Learning-Hyperparameter-Optimizers.pdf)  
27. A/B Testing in Marketing: The Complete Guide to Maximizing Performance \- SevenAtoms, accessed March 7, 2026, [https://www.sevenatoms.com/blog/ab-testing-in-marketing](https://www.sevenatoms.com/blog/ab-testing-in-marketing)  
28. A/B Testing — What it is, examples, and best practices \- Adobe for Business, accessed March 7, 2026, [https://business.adobe.com/blog/basics/learn-about-a-b-testing](https://business.adobe.com/blog/basics/learn-about-a-b-testing)  
29. 6 Real Examples and Case Studies of A/B Testing \- Contentsquare, accessed March 7, 2026, [https://contentsquare.com/guides/ab-testing/examples/](https://contentsquare.com/guides/ab-testing/examples/)  
30. Creativity that converts: how to apply A/B testing to your brand storytelling \- DDigitals, accessed March 7, 2026, [https://ddigitals.net/en/blog/digital-marketing/creativity-that-converts-how-to-apply-test-a-b-to-your-brand-storytelling/](https://ddigitals.net/en/blog/digital-marketing/creativity-that-converts-how-to-apply-test-a-b-to-your-brand-storytelling/)