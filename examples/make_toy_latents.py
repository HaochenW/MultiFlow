"""Create a tiny paired latent dataset for the CLI quickstart."""

from pathlib import Path

import numpy as np


def main() -> None:
    generator = np.random.default_rng(7)
    labels = np.repeat(np.arange(3, dtype=np.int64), 32)
    context = generator.normal(size=(3, 8)).astype(np.float32)
    rna = context[labels] + 0.20 * generator.normal(size=(labels.size, 8))
    atac = context[labels] + 0.20 * generator.normal(size=(labels.size, 8))
    output = Path("toy_paired_latents.npz")
    np.savez_compressed(
        output,
        rna=rna.astype(np.float32),
        atac=atac.astype(np.float32),
        labels=labels,
    )
    print(f"saved {output.resolve()}")


if __name__ == "__main__":
    main()
