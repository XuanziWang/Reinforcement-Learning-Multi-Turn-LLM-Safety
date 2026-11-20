
from datasets import load_dataset
import pandas as pd
import json
from pathlib import Path

# Load MTID
benign = load_dataset("Graph-COM/MTID", "benign")
harmful = load_dataset("Graph-COM/MTID", "harmful")


def preprocess_data(benign_data, harmful_data):
    # Convert to DataFrames
    df_benign = benign_data.to_pandas()
    df_harmful = harmful_data.to_pandas()

    # Add intent labels
    df_benign["intent"] = "benign"
    df_harmful["intent"] = "harmful"

    # Combine both datasets
    df = pd.concat([df_benign, df_harmful], ignore_index=True)

    # Keep relevant columns
    df_processed = df[[
        "sample_index",
        "rollout_id",
        "intent",
        "meta_intent",
        "conversation"
    ]].copy()

    # Create scenario ID
    df_processed["scenario_id"] = (
        df_processed["intent"] + "_" +
        df_processed["sample_index"].astype(str)
    )

    # Convert messages to dictionaries
    df_processed["messages"] = df_processed["conversation"].apply(
        lambda conv: [
            json.loads(msg) if isinstance(msg, str) else dict(msg)
            for msg in conv
        ]
    )

    # Keep only role and content
    df_processed["messages"] = df_processed["messages"].apply(
        lambda conv: [
            {"role": msg["role"], "content": msg["content"]}
            for msg in conv
        ]
    )

    # Extract user prompts
    df_processed["user_prompts"] = df_processed["messages"].apply(
        lambda conv: [
            msg["content"] for msg in conv
            if msg["role"] == "user"
        ]
    )

    # Extract assistant responses
    df_processed["assistant_responses"] = df_processed["messages"].apply(
        lambda conv: [
            msg["content"] for msg in conv
            if msg["role"] == "assistant"
        ]
    )

    return df_processed.drop(columns=["conversation"])


# Create output folder
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Process each split
for split in benign.keys():
    df_processed = preprocess_data(
        benign[split],
        harmful[split]
    )

    # Save as JSONL
    output_file = output_dir / f"mtid_{split}.jsonl"

    df_processed.to_json(
        output_file,
        orient="records",
        lines=True,
        force_ascii=False
    )

    print(f"{split}: {len(df_processed)} conversations saved")


# Check the saved datasets
for split in benign.keys():
    file_path = output_dir / f"mtid_{split}.jsonl"

    df_check = pd.read_json(file_path, lines=True)

    print(f"\n{split.upper()}")
    print("Shape:", df_check.shape)
    print("Unique scenarios:", df_check["scenario_id"].nunique())
    print("Intent counts:")
    print(df_check["intent"].value_counts())


scenario_sets = {}

for split in benign.keys():
    df_check = pd.read_json(
        output_dir / f"mtid_{split}.jsonl",
        lines=True
    )
    scenario_sets[split] = set(df_check["scenario_id"])

splits = list(scenario_sets)

for i in range(len(splits)):
    for j in range(i + 1, len(splits)):
        overlap = scenario_sets[splits[i]] & scenario_sets[splits[j]]

        print(
            f"{splits[i]} vs {splits[j]}: "
            f"{len(overlap)} overlapping scenario IDs"
        )
