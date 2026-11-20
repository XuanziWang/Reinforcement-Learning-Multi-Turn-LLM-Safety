
import pandas as pd
from pathlib import Path

data_dir = Path("data/processed")

for split in ["train", "validation", "test"]:
    # Load processed conversations
    df = pd.read_json(
        data_dir / f"mtid_{split}.jsonl",
        lines=True
    )

    # Keep one representative rollout per scenario
    df_seeds = (
        df.sort_values("rollout_id")
        .drop_duplicates(subset=["scenario_id"])
        .copy()
    )

    # Extract the first user message
    df_seeds["initial_prompt"] = df_seeds["user_prompts"].apply(
        lambda prompts: prompts[0]
    )

    # Keep relevant columns
    df_seeds = df_seeds[[
        "scenario_id",
        "intent",
        "meta_intent",
        "initial_prompt"
    ]]

    # Save the seeds
    output_file = data_dir / f"rl_seeds_{split}.jsonl"

    df_seeds.to_json(
        output_file,
        orient="records",
        lines=True,
        force_ascii=False
    )

    print(f"{split}: {len(df_seeds)} RL seeds saved")

print("\nExample RL seed:")
print(df_seeds.iloc[0].to_dict())