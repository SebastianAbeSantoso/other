import os
import json
import time
import shutil
import requests
import pandas as pd

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "xiaomi/mimo-v2.6-pro"

RESULTS_FILE = "results.xlsx"
GROUND_TRUTHS_FILE = "ground_truths.xlsx"
OUTPUT_FILE = "pilot_evaluation.csv"
BACKUP_FILE = "pilot_evaluation_backup.csv"

if not API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing. Set it in your terminal first."
    )

# ============================================================
# LOAD AND VALIDATE INPUT FILES
# ============================================================

results = pd.read_excel(RESULTS_FILE)
ground_truths = pd.read_excel(GROUND_TRUTHS_FILE)

required_gt = {
    "prompt_id",
    "category",
    "ground_truth",
    "acceptance_criteria",
    "critical_errors",
}

required_results = {
    "record_id",
    "prompt_id",
    "model_name",
    "run_id",
    "prompt",
    "response",
}

if not required_gt.issubset(ground_truths.columns):
    raise ValueError(
        f"Missing ground-truth columns: "
        f"{required_gt - set(ground_truths.columns)}"
    )

if not required_results.issubset(results.columns):
    raise ValueError(
        f"Missing results columns: "
        f"{required_results - set(results.columns)}"
    )

if not ground_truths["prompt_id"].is_unique:
    raise ValueError("Ground-truth prompt_id values must be unique.")

if results["record_id"].duplicated().any():
    raise ValueError("results.xlsx contains duplicate record_id values.")

if not results["prompt_id"].isin(ground_truths["prompt_id"]).all():
    raise ValueError("Some results have no matching ground truth.")

evaluated = results.merge(
    ground_truths,
    on="prompt_id",
    how="left",
    validate="many_to_one",
    suffixes=("", "_gt"),
)

# ============================================================
# RECREATE THE SAME 30-RESPONSE PILOT SAMPLE
# ============================================================

prompt_sample = (
    evaluated[["prompt_id", "category"]]
    .drop_duplicates()
    .groupby("category", group_keys=False)
    .sample(n=5, random_state=42)
)

sample = (
    evaluated.merge(
        prompt_sample[["prompt_id"]],
        on="prompt_id",
        how="inner",
    )
    .groupby("prompt_id", group_keys=False)
    .sample(n=1, random_state=42)
    .reset_index(drop=True)
)

sample["record_id"] = sample["record_id"].astype(str)

if sample["record_id"].duplicated().any():
    raise ValueError("The pilot sample contains duplicate record IDs.")

if len(sample) != 30:
    raise ValueError(
        f"Expected 30 pilot responses, but found {len(sample)}. "
        "Check your input files and sampling configuration."
    )

print("\nPilot sample by category:")
print(sample.groupby("category").size())
print("Total pilot responses:", len(sample))

# ============================================================
# BACK UP THE EXISTING CSV ONCE; NEVER OVERWRITE THE BACKUP
# ============================================================

if os.path.exists(OUTPUT_FILE) and not os.path.exists(BACKUP_FILE):
    shutil.copy2(OUTPUT_FILE, BACKUP_FILE)
    print(f"Backup created: {BACKUP_FILE}")

# ============================================================
# INITIALIZE ALL 30 ROWS AND RESTORE SAVED RESULTS
# ============================================================

metadata_columns = [
    "record_id",
    "prompt_id",
    "model_name",
    "run_id",
]

evaluation_columns = [
    "quality_score",
    "accepted",
    "evaluation_reason",
    "judge_model",
    "error",
]

# Start with all 30 expected records, not an empty results list.
pilot_df = sample[metadata_columns].copy()

for column in evaluation_columns:
    pilot_df[column] = None

# Restore existing results by record_id.
if os.path.exists(OUTPUT_FILE):
    old_df = pd.read_csv(OUTPUT_FILE, dtype={"record_id": str})

    if "record_id" not in old_df.columns:
        raise ValueError(
            f"{OUTPUT_FILE} has no record_id column. "
            "Your existing results have been left untouched."
        )

    old_df["record_id"] = old_df["record_id"].astype(str)
    old_df = old_df.drop_duplicates("record_id", keep="last")
    old_lookup = old_df.set_index("record_id")

    for column in evaluation_columns:
        if column in old_lookup.columns:
            saved_values = pilot_df["record_id"].map(old_lookup[column])
            pilot_df[column] = saved_values

    # Restore existing metadata too, where available.
    for column in ["prompt_id", "model_name", "run_id"]:
        if column in old_lookup.columns:
            saved_values = pilot_df["record_id"].map(old_lookup[column])
            pilot_df[column] = saved_values.where(
                saved_values.notna(), pilot_df[column]
            )

# Save the complete 30-row structure BEFORE making API calls.
pilot_df.to_csv(OUTPUT_FILE, index=False)

# ============================================================
# JUDGE FUNCTION
# ============================================================

def judge_response(row):
    raw_response = row["response"]

    response = (
        str(raw_response)
        if pd.notna(raw_response)
        else ""
    )

    # An empty original model response is a valid evaluation outcome.
    if not response.strip():
        return {
            "quality_score": 0,
            "accepted": 0,
            "evaluation_reason": "Empty model response.",
            "judge_model": JUDGE_MODEL,
            "error": None,
        }

    prompt = f"""
Evaluate this LLM response against the prompt, ground truth,
acceptance criteria, and critical errors.

Judge correctness, relevance, completeness, and task satisfaction.
Accept alternative valid approaches. Do not reward verbosity alone.

ORIGINAL PROMPT:
{row["prompt"]}

GROUND TRUTH:
{row["ground_truth"]}

ACCEPTANCE CRITERIA:
{row["acceptance_criteria"]}

CRITICAL ERRORS:
{row["critical_errors"]}

MODEL RESPONSE:
{response}

ORIGINAL MODEL COMPLETION STATUS:
{row.get("completion_status", "unknown")}

SCORING:
90-100: Correct, complete, and satisfies the prompt.
70-89: Mostly correct, with minor omissions.
40-69: Partially correct, with significant omissions or errors.
1-39: Mostly incorrect or unhelpful.
0: Empty, irrelevant, or fundamentally incorrect.

Set accepted to 1 only when essential acceptance criteria are
satisfied and no critical error is present. Otherwise set it to 0.

Return a JSON object with exactly these keys:
{{
    "quality_score": integer from 0 to 100,
    "accepted": 0 or 1,
    "evaluation_reason": "brief explanation"
}}
"""

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an impartial evaluator. "
                    "Return only the requested JSON object, "
                    "with no extra text. Keep evaluation_reason "
                    "concise, ideally 2-4 sentences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": 2048,
    }

    last_error = None

    for attempt in range(3):
        try:
            api_response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )

            api_response.raise_for_status()
            data = api_response.json()

            if not data.get("choices"):
                raise ValueError(f"No choices returned: {data}")

            message = data["choices"][0].get("message", {})
            content = message.get("content")

            if not isinstance(content, str) or not content.strip():
                finish_reason = data["choices"][0].get("finish_reason")
                raise ValueError(
                    f"Judge returned no text content; "
                    f"finish_reason={finish_reason}"
                )

            content = content.strip()

            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0].strip()

            parsed = json.loads(content)

            score = parsed["quality_score"]
            accepted = parsed["accepted"]
            reason = parsed["evaluation_reason"]

            if (
                isinstance(score, bool)
                or not isinstance(score, int)
                or not 0 <= score <= 100
            ):
                raise ValueError("Invalid quality_score.")

            if (
                isinstance(accepted, bool)
                or not isinstance(accepted, int)
                or accepted not in (0, 1)
            ):
                raise ValueError("Invalid accepted value.")

            if not isinstance(reason, str):
                raise ValueError("Invalid evaluation_reason.")

            return {
                "quality_score": score,
                "accepted": accepted,
                "evaluation_reason": reason,
                "judge_model": JUDGE_MODEL,
                "error": None,
            }

        except Exception as e:
            last_error = e
            retry_token_limits = [2048, 3072, 4096]

            for attempt in range(3):
                payload["max_tokens"] = retry_token_limits[attempt]

                wait_seconds = 2 ** attempt
                print(
                    f"  Attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait_seconds}s..."
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Judge failed after 3 attempts: {last_error}"
    )

# ============================================================
# RESUMABLE EVALUATION
# ============================================================

def is_successful(row):
    score = pd.to_numeric(
        pd.Series([row["quality_score"]]), errors="coerce"
    ).iloc[0]

    accepted = pd.to_numeric(
        pd.Series([row["accepted"]]), errors="coerce"
    ).iloc[0]

    has_error = pd.notna(row["error"]) and str(row["error"]).strip() != ""

    return pd.notna(score) and pd.notna(accepted) and not has_error


completed = sum(
    is_successful(row)
    for _, row in pilot_df.iterrows()
)

print("\nPreviously successful evaluations:", completed)
print("Evaluations still needing work:", len(sample) - completed)

for _, row in sample.iterrows():
    record_id = str(row["record_id"])
    mask = pilot_df["record_id"].astype(str) == record_id

    if not mask.any():
        raise ValueError(f"Missing initialized record: {record_id}")

    current = pilot_df.loc[mask].iloc[0]

    if is_successful(current):
        print(f"SKIP already successful: {record_id}")
        continue

    print(f"EVALUATING: {record_id}")

    try:
        result = judge_response(row)

    except Exception as e:
        result = {
            "quality_score": None,
            "accepted": None,
            "evaluation_reason": None,
            "judge_model": JUDGE_MODEL,
            "error": str(e),
        }

        print(f"  FAILED: {e}")

    for column, value in result.items():
        pilot_df.loc[mask, column] = value

    # Persist immediately. Never write only the current batch.
    pilot_df.to_csv(OUTPUT_FILE, index=False)
    time.sleep(1)

# ============================================================
# FINAL SUMMARY
# ============================================================

pilot_df = pd.read_csv(OUTPUT_FILE)

successful_mask = pilot_df.apply(is_successful, axis=1)
successful = pilot_df[successful_mask]
failed = pilot_df[~successful_mask]

print("\n================ FINAL SUMMARY ================")
print("Total pilot responses:", len(pilot_df))
print("Successful evaluations:", len(successful))
print("Failed evaluations:", len(failed))

if not successful.empty:
    print(
        "Mean quality score:",
        pd.to_numeric(
            successful["quality_score"], errors="coerce"
        ).mean(),
    )
    print(
        "Acceptance rate:",
        pd.to_numeric(
            successful["accepted"], errors="coerce"
        ).mean(),
    )

if not failed.empty:
    print("\nRecords still needing evaluation:")
    print(failed[["record_id", "error"]].to_string(index=False))

print(f"\nResults saved to: {OUTPUT_FILE}")
print(f"Backup saved to: {BACKUP_FILE}")
