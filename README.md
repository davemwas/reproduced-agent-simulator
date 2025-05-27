Reproducing AgentSimulator Evaluation Results

Objective

The goal of this assignment was to reproduce the evaluation results presented in the AgentSimulator paper by Kirchdorfer et al. This involved assessing how well different simulation configurations (e.g., FP_Auto, PN_Orch, etc.) replicated real-world event logs across nine public datasets.

⸻

Datasets Used

The reproduction used nine real-world process mining datasets, including:

	•	Loan
	•	P2P
	•	CVS
	•	C1000, C2000
	•	ACR
	•	Production
	•	BPIC_2012W
	•	BPIC_2017W

For each dataset. [Author's Drive With Dataset](https://drive.google.com/file/d/10OcbxF9hSoiItb8zAb3W5oxKTiNkaXHg/view?usp=sharing):

	•	The real logs (test_preprocessed.csv) were sourced from Evaluation_1
	•	The simulated logs (10 per configuration) were sourced from Evaluation_2

⸻

Evaluation Method

Each simulation configuration (such as FP_Auto) produced 10 simulated log files.
Each file was compared to the corresponding real test log using the following five metrics:

Metric	Description
NGD	N-Gram Distance – sequence similarity of activities

AED	Absolute Event Distribution – frequency of activity labels

CED	Circadian Event Distribution – time-of-day patterns

RED	Relative Event Distribution – timing of events within each case

CTD	Cycle Time Distribution – duration similarity (measured as RMSE)

Each metric was calculated for every simulation run, and the final score was the average across all 10 runs.

⸻

Tooling

A custom Python script was implemented to:

	•	Loop through all datasets and configurations
	•	Compute all five metrics by comparing each simulated log to the real test log
	•	Average the metric scores across the 10 runs
	•	Output a structured, single-sheet Excel file with:
	•	Rows: Metric and Method
	•	Columns: Dataset names

The output file is named reproduction_results.xlsx.

⸻

Results

The final table summarizes the averaged results for each configuration across all datasets.
This structure replicates the format used in the original paper and [GitHub repository.](https://github.com/lukaskirchdorfer/agentsimulator)

In most cases, the reproduced values were close to those reported by the author, confirming the validity of the experimental setup. Minor differences are expected due to inherent randomness in simulation and environment-specific variations.

⸻

Conclusion

This reproduction confirms that:

	•	The evaluation pipeline and metric calculations were correctly implemented
	•	The original findings of the paper are largely reproducible
	•	AgentSimulator’s behavior is consistent across independent evaluations

This task also reinforced an understanding of simulation evaluation, reproducibility in computational research, and structured benchmarking using real-world event logs.