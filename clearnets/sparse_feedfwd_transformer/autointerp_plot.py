import json
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.io as pio

pio.kaleido.scope.mathjax = None  # https://github.com/plotly/plotly.py/issues/3469


def parse_score_file(file_path):
    with open(file_path, "r") as f:
        content = f.read().strip()

    try:
        # Try direct JSON parsing first 
        if content.startswith("[") and content.endswith("]"):
            data = json.loads(content)
        else:
            # Try XML format if direct JSON fails
            start = content.find("<document_content>") + len("<document_content>")
            end = content.find("</document_content>")
            if start == -1 or end == -1:
                print(f"Could not parse {file_path.name} - invalid format")
                return None
            data = json.loads(content[start:end])

        # Parse into DataFrame
        df = pd.DataFrame(
            [
                {
                    "text": "".join(segment["str_tokens"]),
                    "distance": segment["distance"],
                    "ground_truth": segment["ground_truth"],
                    "prediction": segment["prediction"],
                    "probability": segment["probability"],
                    "correct": segment["correct"],
                    "activations": segment["activations"],
                    "highlighted": segment["highlighted"],
                }
                for segment in data
            ]
        )

        return df

    except json.JSONDecodeError as e:
        print(f"Error parsing {file_path.name}: {e}")
        return None


def build_df(path: Path):
    accuracies = []
    probabilities = []
    score_types = []
    latent_type = []
    feature_idx = []

    for type in ["sparse", "SAE", "Transcoder"]:
        if type == "sparse":
            dir_path = path / f"sparse-8m-max-e=200-esp=15-s=42" / "default"
        else:
            dir_path = path / f"mlp=1024-dense-8m-max-e=200-esp=15-s=42" / type.lower()

        for score_type in ["fuzz", "detection"]:
            for score_file in (dir_path / score_type).glob("*.txt"):

                df = parse_score_file(score_file)
                if df is None:
                    continue

                # Calculate the accuracy and cross entropy loss for this example
                latent_type.append(type)
                score_types.append(score_type)
                feature_idx.append(int(score_file.stem.split("feature")[-1]))
                accuracies.append(df["correct"].mean())
                probabilities.append(df["probability"].mean())

    df = pd.DataFrame(
        {
            "latent_type": latent_type,
            "score_type": score_types,
            "feature_idx": feature_idx,
            "accuracy": accuracies,
            "probabilities": probabilities,
        }
    )

    # Plot histograms of cross entropy loss with each score type being a different subplot
    # and each latent type being a different color
    out_path = Path("images")
    out_path.mkdir(parents=True, exist_ok=True)

    for score_type in df["score_type"].unique():
        fig = px.histogram(
            df[df["score_type"] == score_type],
            x="probabilities",
            color="latent_type",
            barmode="overlay",
            title=f"Probability Distribution - {score_type}",
            nbins=100,
        )
        fig.write_image(out_path / f"autointerp_probabilities_{score_type}.pdf", format="pdf")

        fig = px.histogram(
            df[df["score_type"] == score_type],
            x="accuracy",
            color="latent_type",
            barmode="overlay",
            title=f"Accuracy Distribution - {score_type}",
            nbins=100,
        )
        fig.write_image(out_path / f"autointerp_accuracies_{score_type}.pdf", format="pdf")

    df.to_csv("autointerp_results.csv", index=False)

    # Print the mean accuracy and probability for each score type and latent type
    for score_type in df["score_type"].unique():
        for latent_type in df["latent_type"].unique():
            print(f"{score_type} - {latent_type}:")
            print(
                f"  Mean accuracy: {df[(df['score_type'] == score_type) & (df['latent_type'] == latent_type)]['accuracy'].mean()}"
            )
            print(
                f"  Mean probability: {df[(df['score_type'] == score_type) & (df['latent_type'] == latent_type)]['probabilities'].mean()}"
            )


if __name__ == "__main__":
    path = Path.cwd() / "results/scores/tinystories"
    
    build_df(path)
