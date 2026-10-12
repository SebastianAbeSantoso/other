
import os
import json
import time
import shutil
import requests
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "xiaomi/mimo-v2.6-pro"

RESULTS_FILE = "results.xlsx"
GROUND_TRUTHS_FILE = "ground_truths.xlsx"
OUTPUT_FILE = "full_evaluation.csv"
BACKUP_FILE = "full_evaluation_backup.csv"

EXPECTED_RECORDS = 1200
RETRY_TOKEN_LIMITS = [2048, 3072, 4096]
MAX_ATTEMPTS = len(RETRY_TOKEN_LIMITS)

if not API_KEY or API_KEY == "x":
    raise ValueError(
        'Set your real API key first using: '
        '$env:OPENROUTER_API_KEY="your-key"'
    )

# ============================================================
# LOAD AND VALIDATE INPUT FILES
# ============================================================

results = pd.read_excel(RESULTS_FILE)
ground_truths = pd.read_excel(GROUND_TRUTHS_FILE)

required_gt = {
    "prompt_id", "category", "ground_truth",
    "acceptance_criteria", "critical_errors"
}
required_results = {
    "record_id", "prompt_id", "model_name",
    "run_id", "prompt", "response"
}

if not required_gt.issubset(ground_truths.columns):
    raise ValueError(
        f"Missing ground-truth columns: "
        f"{required_gt - set(ground_truths.columns)}"
    )

if not required_results.issubset(results.columns):
    raise ValueError(
        f"Missing result columns: "
        f"{required_results - set(results.columns)}"
    )

if not ground_truths["prompt_id"].is_unique:
    raise ValueError("Ground-truth prompt_id values must be unique.")

if results["record_id"].isna().any():
    raise ValueError("Some result records have missing record_id values.")

results["record_id"] = results["record_id"].astype(str)

if results["record_id"].duplicated().any():
    raise ValueError("Duplicate record_id values found in results.xlsx.")

if not results["prompt_id"].isin(ground_truths["prompt_id"]).all():
    raise ValueError("Some results have no matching ground truth.")

evaluated = results.merge(
    ground_truths,
    on="prompt_id",
    how="left",
    validate="many_to_one",
    suffixes=("", "_gt"),
)

if len(evaluated) != EXPECTED_RECORDS:
    raise ValueError(
        f"Expected {EXPECTED_RECORDS} records, "
        f"but found {len(evaluated)}. Check your input file."
    )

# IMPORTANT: Evaluate every record, not the 30-row pilot sample.
sample = evaluated.copy().reset_index(drop=True)
sample["record_id"] = sample["record_id"].astype(str)

print("Records to evaluate:", len(sample))
print("Unique models:", sample["model_name"].nunique())
print("Unique prompts:", sample["prompt_id"].nunique())

# ============================================================
# BACK UP EXISTING FULL RESULTS, IF PRESENT
# ============================================================

if os.path.exists(OUTPUT_FILE) and not os.path.exists(BACKUP_FILE):
    shutil.copy2(OUTPUT_FILE, BACKUP_FILE)
    print(f"Existing full results backed up to {BACKUP_FILE}")

# ============================================================
# INITIALIZE ALL 1,200 ROWS AND RESTORE SAVED RESULTS
# ============================================================

metadata_columns = [
    "record_id", "prompt_id", "model_name", "run_id"
]

evaluation_columns = [
    "quality_score", "accepted", "evaluation_reason",
    "judge_model", "error"
]

pilot_df = sample[metadata_columns].copy()

for column in evaluation_columns:
    pilot_df[column] = None

if os.path.exists(OUTPUT_FILE):
    old_df = pd.read_csv(
        OUTPUT_FILE,
        dtype={"record_id": str}
    )

    if "record_id" not in old_df.columns:
        raise ValueError(
            f"{OUTPUT_FILE} has no record_id column. "
            "Existing results have not been overwritten."
        )

    old_df = old_df.drop_duplicates("record_id", keep="last")
    old_df = old_df.set_index("record_id")

    for column in evaluation_columns:
        if column in old_df.columns:
            saved_values = pilot_df["record_id"].map(old_df[column])
            pilot_df[column] = saved_values

# Save the complete 1,200-row structure before API calls.
def save_results(df):
    temp_file = OUTPUT_FILE + ".tmp"
    df.to_csv(temp_file, index=False)
    os.replace(temp_file, OUTPUT_FILE)

save_results(pilot_df)

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

    # Empty original response is a valid result, not an API failure.
    if not response.strip():
        return {
            "quality_score": 0,
            "accepted": 0,
            "evaluation_reason": "Empty model response.",
            "judge_model": JUDGE_MODEL,
            "error": None,
        }

    prompt = f"""
Evaluate this LLM response against the original prompt, ground truth,
acceptance criteria, and critical errors.

Assess correctness, relevance, completeness, and task satisfaction.
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

QUALITY SCORE:
90-100: Correct, complete, and satisfies the prompt.
70-89: Mostly correct, with minor omissions.
40-69: Partially correct, with significant omissions or errors.
1-39: Mostly incorrect or unhelpful.
0: Empty, irrelevant, or fundamentally incorrect.

Set accepted to 1 only when essential acceptance criteria are met
and no critical error is present. Otherwise, set accepted to 0.

Keep evaluation_reason concise, ideally 2-4 sentences.

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
                    "You are an impartial evaluator. Follow the supplied "
                    "rubric and return only the requested JSON object. "
                    "Do not include analysis outside the JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }

    last_error = None

    # Each attempt makes a NEW API request with a larger token budget.
    for attempt, token_limit in enumerate(RETRY_TOKEN_LIMITS):
        payload["max_tokens"] = token_limit

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

            choice = data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content")

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    "No judge text content; "
                    f"finish_reason={choice.get('finish_reason')}"
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
                raise ValueError("quality_score must be an integer from 0 to 100.")

            if (
                isinstance(accepted, bool)
                or not isinstance(accepted, int)
                or accepted not in (0, 1)
            ):
                raise ValueError("accepted must be integer 0 or 1.")

            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("evaluation_reason must be a non-empty string.")

            return {
                "quality_score": score,
                "accepted": accepted,
                "evaluation_reason": reason.strip(),
                "judge_model": JUDGE_MODEL,
                "error": None,
            }

        except Exception as e:
            last_error = e

            if attempt < MAX_ATTEMPTS - 1:
                wait_seconds = 2 ** attempt
                print(
                    f"  Attempt {attempt + 1} failed "
                    f"(token budget {token_limit}): {e}"
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Judge failed after {MAX_ATTEMPTS} attempts: {last_error}"
    )

# ============================================================
# SUCCESS CHECK
# ============================================================

def is_successful(row):
    score = pd.to_numeric(
        pd.Series([row["quality_score"]]), errors="coerce"
    ).iloc[0]

    accepted = pd.to_numeric(
        pd.Series([row["accepted"]]), errors="coerce"
    ).iloc[0]

    error = row["error"]
    has_error = pd.notna(error) and str(error).strip() != ""

    return (
        pd.notna(score)
        and pd.notna(accepted)
        and not has_error
    )

# ============================================================
# RESUMABLE FULL EVALUATION
# ============================================================

completed = sum(
    is_successful(row)
    for _, row in pilot_df.iterrows()
)

print("\nPreviously successful:", completed)
print("Still needing evaluation:", len(sample) - completed)

for index, row in sample.iterrows():
    record_id = str(row["record_id"])
    mask = pilot_df["record_id"].astype(str) == record_id

    if not mask.any():
        raise ValueError(f"Record missing from output: {record_id}")

    current = pilot_df.loc[mask].iloc[0]

    if is_successful(current):
        continue

    print(
        f"[{index + 1}/{len(sample)}] "
        f"Evaluating {record_id}"
    )

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

    # Save after EVERY record, including failures.
    save_results(pilot_df)

    time.sleep(1)

# ============================================================
# FINAL SUMMARY
# ============================================================

pilot_df = pd.read_csv(OUTPUT_FILE)

successful_mask = pilot_df.apply(is_successful, axis=1)
successful = pilot_df[successful_mask]
failed = pilot_df[~successful_mask]

print("\n=============== FINAL SUMMARY ===============")
print("Total records:", len(pilot_df))
print("Successful evaluations:", len(successful))
print("Failed evaluations:", len(failed))

if not successful.empty:
    scores = pd.to_numeric(
        successful["quality_score"], errors="coerce"
    )
    accepts = pd.to_numeric(
        successful["accepted"], errors="coerce"
    )

    print("Mean quality score:", scores.mean())
    print("Acceptance rate:", accepts.mean())

if not failed.empty:
    print("\nFailed records to retry on the next run:")
    print(
        failed[["record_id", "prompt_id", "model_name", "error"]]
        .to_string(index=False)
    )

print(f"\nResults saved to: {OUTPUT_FILE}")
print("Successful rows will be skipped on the next run.")
