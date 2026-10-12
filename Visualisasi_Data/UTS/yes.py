import pandas as pd

pilot_df = pd.read_csv("pilot_evaluation.csv")

accepted = pilot_df[pilot_df["evaluation_reason"].notna()]

print(
    accepted[
        ["record_id", "prompt_id", "model_name", "evaluation_reason", "quality_score"]
    ].to_string(index=False)
)