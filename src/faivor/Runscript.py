import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt

#Add src to path
sys.path.insert(0,str(Path(__file__).parent/ "src"))

from faivor.Calibration import Calibration

input_data = pd.read_csv("data/Supplemental_Data_S1_val2.csv", sep =";", decimal=",")
with open("data/prediction_faivor.json", "r", encoding="utf-8") as f:
    preds = json.load(f)
predictions = preds["prediction_2"]

def calibration_results():
    print("\n" + "=" * 70)
    print("=" * 70)

    print("\nDataset info:")
    print(f" Shape : {input_data.shape}")
    print(f" Columns : {list(input_data.columns)}")
    print(input_data.head())

    outcome_label='morbidityyesorno'
    if outcome_label in input_data.columns:
       outcomes = input_data[outcome_label].to_numpy()
       expected_outputs =[{outcome_label:int(y)} for y in outcomes]
       outcome_col = outcome_label
    else:
        print("outcome label not found")
        print(f"Available outcome labels: {list(input_data.columns)}")
        return

    feature_data = input_data.drop(columns=[outcome_label,'ID','newmorbidityyesorno','morbidityyesorno',
                                   'ppoFEV1_0','FVC_0','cardiac_comorbidity','FVC','ACCI','CKD','CAD','CVD'],errors= 'ignore')

    feature_data = feature_data.select_dtypes(include=[np.number])

    calibration = Calibration(
        predictions = predictions,
        expected_outputs =expected_outputs,
        outcome_col = outcome_col,
        input_data = feature_data,
    )

    calibration.plot_all()
    calibration.plot_calibration_curve()
    calibration.plot_threshold_analysis_all_models()
    plt.show()

    metrics = calibration.compute_classification_metrics(threshold=0.1)
    metrics_df = pd.DataFrame(metrics).T
    print(f"\nCalibration metrics :")
    print(metrics_df.to_string())

    summary = calibration.get_calibration_summary()

    print("\n" + "=" * 70)
    print("Closed testing Looping Results")
    print("-" * 70)

    model_selection = summary["model_selection"]
    selected = model_selection["selected_model"]

    stats = model_selection["statistics"]

    print(f"\nSelected model: {selected.upper()}")
    print("\nModel Selection Statistics:")
    print(f" Test 1 P-value: {stats['test1_p_value']:.4f} (M0 vs M3)")
    print(f" Test 2 P-value: {stats['test2_p_value']:.4f} (M1 vs M3)")
    print(f" Test 3 P-value: {stats['test3_p_value']:.4f} (M2 vs M3)")

    print(f" Test 1 Result: {stats['test1_result']}")
    print(f" Test 2 Result: {stats['test2_result']}")
    print(f" Test 3 Result: {stats['test3_result']}")

    print("\nLog-Likelihoods:")
    lls = summary["log_likelihoods"]
    for model, ll in lls.items():
        print(f"  {model.upper()}: {ll:.4f}")

    print("\nROC AUC Scores:")
    for model,metrics in summary['roc_metrics'].items():
        if 'auc' in metrics:
            print(f"  {model.upper()}: {metrics['auc']:.4f}")

    return summary

def calibration_summary():
    outcome_col='morbidityyesorno'
    covariate_columns=[
        "Sex",
        "age",
        "thoracotomy",
        "extended_resection",
        "ppoFEV1"
    ]
    expected_outputs= input_data[[outcome_col]].to_dict(orient="records")
    feature_data =input_data[covariate_columns].copy()
    calibration = Calibration(
        predictions=predictions,
        expected_outputs=expected_outputs,
        outcome_col=outcome_col,
        input_data=feature_data,
    )
    print(calibration.data.columns.to_list())
    results_table = calibration.get_results_table(covariate_columns=covariate_columns)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)
    print("\nCalibration summary table:\n")
    print(results_table.to_string(index=False))

if __name__ == "__main__":

    calibration_summary()
    calibration_results()



