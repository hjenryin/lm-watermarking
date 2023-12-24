from matplotlib import pyplot as plt
from utils.io import load_jsonlines
from datasets import Dataset

# ppl_file = load_jsonlines(
#     'run_outputs/opt1.3b_N500_T200_eval/gen_table_w_metrics.jsonl')
other_file = load_jsonlines(
    'run_outputs/opt1.3b_N500_T200_eval/gen_table_w_metrics.jsonl')
# ppl_file = Dataset.from_list(ppl_file)
other_file = Dataset.from_list(other_file)

# %%
# ppl_file = ppl_file.filter(lambda x: x['w_wm_output_length'] > 10)
other_file = other_file.filter(lambda x: x['w_wm_output_length'] > 10)
# %%
cols = ["w_wm_output_z_score", "w_wm_output_p_value",
        "w_wm_output_winmax-1_p_value", "w_wm_output_winmax-1_z_score", "w_wm_output_ppl", "w_wm_output_attacked_ppl", "w_wm_output_attacked_baseline_ppl", "w_wm_output_vs_w_wm_output_attacked_p_sp", "w_wm_output_vs_w_wm_output_attacked_baseline_p_sp", "w_wm_output_attacked_log_diversity", "w_wm_output_attacked_baseline_log_diversity", "w_wm_output_log_diversity", "w_wm_output_attacked_winmax-1_z_score", "w_wm_output_attacked_winmax-1_p_value", "w_wm_output_attacked_baseline_winmax-1_z_score", "w_wm_output_attacked_baseline_winmax-1_p_value", "w_wm_output_attacked_z_score", "w_wm_output_attacked_p_value", "w_wm_output_attacked_baseline_z_score", "w_wm_output_attacked_baseline_p_value"
        ]

for col in cols:
    if col in other_file.features:
        print(col, sum(other_file[col])/len(other_file))
# %%
