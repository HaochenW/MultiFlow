# Security policy

## Reporting a vulnerability

Do not report suspected vulnerabilities, leaked credentials, private paths,
or restricted data in a public issue. Use the repository's private security
advisory channel once it is enabled, or contact the maintainers privately using
the institutional contact listed in the public repository.

Include the affected version, a minimal reproduction, impact, and any proposed
mitigation. Do not attach real single-cell data or credentials.

## Checkpoint safety

Only load checkpoints from trusted sources. MultiFlow requests PyTorch's
weights-only loader when supported, but older PyTorch releases do not provide
the same protection. Run untrusted artifacts in an isolated environment and
upgrade to a supported PyTorch version before loading published checkpoints.

## Supported versions

This is a pre-release package. Security fixes will target the latest published
version after public releases begin; no older version is currently supported.
