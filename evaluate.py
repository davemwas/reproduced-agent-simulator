import pandas as pd
import numpy as np
import os
from collections import Counter
from sklearn.metrics import mean_squared_error
import math
from scipy.stats import wasserstein_distance
from statistics import mean
import datetime

# Metric Calculations
def discretize_to_hour(seconds):
    return int(seconds // 3600)


def compute_ctd_author(real_df, sim_df, bin_size_minutes=60):
    bin_size = datetime.timedelta(minutes=bin_size_minutes)

    def compute_cycle_times(df):
        df['start'] = pd.to_datetime(df['start_timestamp'], format='mixed')
        df['end'] = pd.to_datetime(df['end_timestamp'], format='mixed')
        cycle_times = []
        for _, group in df.groupby('case_id'):
            duration = group['end'].max() - group['start'].min()
            cycle_times.append(duration)
        return cycle_times

    real_ct = compute_cycle_times(real_df.copy())
    sim_ct = compute_cycle_times(sim_df.copy())

    
    min_duration = min(real_ct + sim_ct)
    def discretize(ct_list):
        return [math.floor((ct - min_duration) / bin_size) for ct in ct_list]

    real_bins = discretize(real_ct)
    sim_bins = discretize(sim_ct)

    return wasserstein_distance(real_bins, sim_bins)

def _relativize_and_discretize(df, use_start=True, use_end=True):
    df['start'] = pd.to_datetime(df['start_timestamp'], format='mixed')
    df['end'] = pd.to_datetime(df['end_timestamp'], format='mixed')
    results = []
    for _, group in df.groupby('case_id'):
        trace_start = group['start'].min()
        if use_start:
            for t in group['start']:
                diff = (t - trace_start).total_seconds()
                results.append(discretize_to_hour(diff))
        if use_end:
            for t in group['end']:
                diff = (t - trace_start).total_seconds()
                results.append(discretize_to_hour(diff))
    return results

def compute_red_author(real_df, sim_df):
    real_rel = _relativize_and_discretize(real_df.copy(), use_start=True, use_end=True)
    sim_rel = _relativize_and_discretize(sim_df.copy(), use_start=True, use_end=True)
    return wasserstein_distance(real_rel, sim_rel)

def compute_aed_author(real_df, sim_df):
    real_hours = pd.to_datetime(real_df['end_timestamp'], format='mixed').dt.hour
    sim_hours = pd.to_datetime(sim_df['end_timestamp'], format='mixed').dt.hour

    return wasserstein_distance(real_hours, sim_hours)

def _compute_n_grams(df, case_col, activity_col, n):
    ngram_counts = Counter()
    grouped = df.sort_values(by='end_timestamp').groupby(case_col)
    for _, group in grouped:
        activities = group[activity_col].tolist()
        ngrams = zip(*[activities[i:] for i in range(n)])
        ngram_counts.update(ngrams)
    return ngram_counts


def compute_ngd_author(real_df, sim_df, n=3):
    real_ngrams = _compute_n_grams(real_df, 'case_id', 'activity_name', n)
    sim_ngrams = _compute_n_grams(sim_df, 'case_id', 'activity_name', n)

    all_keys = set(real_ngrams.keys()) | set(sim_ngrams.keys())
    real_freqs = [real_ngrams.get(k, 0) for k in all_keys]
    sim_freqs = [sim_ngrams.get(k, 0) for k in all_keys]

    distance = sum(abs(r - s) for r, s in zip(real_freqs, sim_freqs))
    total = sum(real_freqs) + sum(sim_freqs)
    return distance / total if total > 0 else 0.0


def compute_ced_author(real_df, sim_df):
    def discretize(df):
        df['timestamp'] = pd.to_datetime(df['end_timestamp'], format='mixed')
        return pd.DataFrame({
            'hour': df['timestamp'].dt.hour,
            'weekday': df['timestamp'].dt.dayofweek
        })

    real = discretize(real_df)
    sim = discretize(sim_df)

    distances = []
    for day in range(7):
        real_day = real[real['weekday'] == day]['hour']
        sim_day = sim[sim['weekday'] == day]['hour']
        if len(real_day) > 0 and len(sim_day) > 0:
            dist = wasserstein_distance(real_day, sim_day)
            distances.append(dist)
        elif len(real_day) == 0 and len(sim_day) == 0:
            distances.append(0.0)
        else:
            distances.append(23.0)
    return mean(distances)


def extract_traces(df, case_col='case_id', activity_col='activity_name'):
    return df.groupby(case_col)[activity_col].apply(list).tolist()

def compute_ngd(real, simulated, n=2):
    def ngrams(trace, n): return [tuple(trace[i:i+n]) for i in range(len(trace)-n+1)]
    real_ngrams, sim_ngrams = Counter(), Counter()
    for trace in real: real_ngrams.update(ngrams(trace, n))
    for trace in simulated: sim_ngrams.update(ngrams(trace, n))
    all_ngrams = set(real_ngrams) | set(sim_ngrams)
    numerator = sum(abs(real_ngrams[g] - sim_ngrams[g]) for g in all_ngrams)
    denominator = sum(real_ngrams[g] + sim_ngrams[g] for g in all_ngrams)
    return numerator / denominator if denominator else 0.0

def compute_ctd(real_df, sim_df):
    real_times = (pd.to_datetime(real_df['end_timestamp'], format='mixed') - pd.to_datetime(real_df['start_timestamp'], format='mixed')).dt.total_seconds()
    sim_times = (pd.to_datetime(sim_df['end_timestamp'], format='mixed') - pd.to_datetime(sim_df['start_timestamp'], format='mixed')).dt.total_seconds()
    return math.sqrt(mean_squared_error(real_times[:min(len(real_times), len(sim_times))], sim_times[:min(len(real_times), len(sim_times))]))

def compute_aed(real_df, sim_df):
    real_dist = real_df['activity_name'].value_counts(normalize=True)
    sim_dist = sim_df['activity_name'].value_counts(normalize=True)
    all_activities = set(real_dist.index).union(sim_dist.index)
    return sum(abs(real_dist.get(a, 0) - sim_dist.get(a, 0)) for a in all_activities) / 2

def compute_red(real_df, sim_df):
    def rel_pos(df):
        df['timestamp'] = pd.to_datetime(df['end_timestamp'], format='mixed')
        rels = []
        for _, g in df.groupby('case_id'):
            times = g.sort_values('timestamp')['timestamp']
            dur = (times.max() - times.min()).total_seconds()
            if dur > 0:
                rels += [(t - times.min()).total_seconds() / dur for t in times]
        return np.histogram(rels, bins=10, range=(0,1), density=True)[0]
    return np.linalg.norm(rel_pos(real_df.copy()) - rel_pos(sim_df.copy()), ord=1) / 2

def compute_ced(real_df, sim_df):
    real_hours = pd.to_datetime(real_df['end_timestamp'], format='mixed').dt.hour
    sim_hours = pd.to_datetime(sim_df['end_timestamp'], format='mixed').dt.hour
    rh = np.histogram(real_hours, bins=24, range=(0, 24), density=True)[0]
    sh = np.histogram(sim_hours, bins=24, range=(0, 24), density=True)[0]
    return np.linalg.norm(rh - sh, ord=1) / 2

# Evaluation
def evaluate_config(dataset, config, eval1_base, eval2_base):
    real_path = os.path.join(eval1_base, 'agent_simulator', dataset, "test_preprocessed.csv")
    config_path = os.path.join(eval2_base, dataset, config)

    if not os.path.exists(real_path):
        print(f"Skipping {dataset}/{config} — test_preprocessed.csv not found.")
        return None

    real_log = pd.read_csv(real_path)
    real_traces = extract_traces(real_log)

    scores = {'NGD': [], 'AEDD': [], 'CEDD': [], 'REDD': [], 'CTDD': []}
    for i in range(10):
        sim_path = os.path.join(config_path, f'simulated_log_{i}.csv')
        if not os.path.exists(sim_path): continue
        sim_log = pd.read_csv(sim_path)
        sim_traces = extract_traces(sim_log)
        scores['NGD'].append(compute_ngd_author(real_log, sim_log))
        scores['AEDD'].append(compute_aed_author(real_log, sim_log))
        scores['CEDD'].append(compute_ced_author(real_log, sim_log))
        scores['REDD'].append(compute_red_author(real_log, sim_log))
        scores['CTDD'].append(compute_ctd_author(real_log, sim_log))

    return {metric: np.mean(values) for metric, values in scores.items()} if scores['NGD'] else None

def run_batch_evaluation(eval1_base, eval2_base):
    results = []
    datasets = [d for d in os.listdir(eval2_base) if os.path.isdir(os.path.join(eval2_base, d))]
    for dataset in datasets:
        dataset_path = os.path.join(eval2_base, dataset)
        configs = [c for c in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, c)) and c.startswith(('FP', 'LS', 'PN'))]
        for config in configs:
            print(f"Evaluating {dataset} / {config}")
            metrics = evaluate_config(dataset, config, eval1_base, eval2_base)
            if metrics:
                results.append({'Dataset': dataset, 'Config': config, **metrics})
    return pd.DataFrame(results)

def save_single_sheet_excel(df, output_path="reproduction_results.xlsx"):
    melted = df.melt(id_vars=['Dataset', 'Config'], var_name='Metric', value_name='Value')
    pivoted = melted.pivot_table(index=['Metric', 'Config'], columns='Dataset', values='Value').reset_index()
    pivoted = pivoted.round(2)
    pivoted.to_excel(output_path, index=False)
    print(f"Output Excel file saved to: {output_path}")

if __name__ == "__main__":
    eval1_base = "process_science_data/Evaluation_1"
    eval2_base = "process_science_data/Evaluation_2"
    df = run_batch_evaluation(eval1_base, eval2_base)
    save_single_sheet_excel(df)