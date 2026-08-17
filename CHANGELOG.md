# Changelog

All notable changes to MultiFlow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project intends to follow [Semantic Versioning](https://semver.org/)
after its first public release.

## [Unreleased]

### Added

- Source-layout Python package for paired latent flow matching.
- Cell-state, perturbation-conditioned, and concatenation-ablation models.
- Feature-wise latent standardization, ODE sampling, checkpoint utilities, and
  a command-line interface.
- Unit tests and release checks for Python 3.10 through 3.12.
- Paired H5MU training and generated-output contracts with string cell-type
  labels retained in portable checkpoints.
- Version-pinned, checksum-verified download command for the scDiffusion-X
  OpenProblem dataset.
- A concise H5MU-first tutorial and an implementation audit linking the public
  models to the executed research references.

### Changed

- The concise user-facing command is now `multiflow`; the historical
  `multiflow-omics` command remains as a compatibility alias.
- The main CLI no longer uses anonymous NPZ files for training or generation.
- Default midpoint-solver steps now match the research protocols: 100 for
  cell-state generation and 50 for perturbation prediction.
