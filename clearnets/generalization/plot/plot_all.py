from pathlib import Path

from clearnets.plot.plot_mnist_vit import plot_mnist_vit
from clearnets.plot.plot_modular_addition_grok import plot_modular_addition_grok
from clearnets.plot.plot_pythia import plot_pythia

if __name__ == "__main__":
    sae_name = "v2"
    run_identifier = f'mnist_seed_1'
    model_path = Path("data") / "vit_ckpts" / run_identifier / "final.pth"
    sae_path = Path("sae") / "vit" / f"{run_identifier}_{sae_name}"
    out_path = Path(f"workspace") / 'inference' / run_identifier / "inference.pth"
    images_path = Path("images")

    plot_mnist_vit(model_path, out_path, images_path)
    plot_modular_addition_grok()
    plot_pythia()
