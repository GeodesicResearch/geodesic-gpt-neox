# Python UV: The fast, unified package manager transforming Python development

UV has emerged as the most significant advancement in Python package management in a decade, delivering **10-100x faster performance** than pip while unifying a fragmented ecosystem into a single tool. Released in February 2024 by Astral—the team behind the Ruff linter—UV replaces pip, pip-tools, virtualenv, pyenv, and pipx with one Rust-powered binary. With **77,000 GitHub stars** and **28 million monthly downloads**, UV now handles over 10% of all PyPI traffic. For developers building pure Python applications, UV offers transformative speed improvements and a modern "Cargo for Python" experience, though conda remains essential for projects requiring non-Python dependencies like CUDA or specialized numerical libraries.

## Astral's mission to fix Python tooling

UV was created by **Astral**, a company founded in April 2023 by Charlie Marsh with $4 million in seed funding from Accel and notable angels including Guillermo Rauch (Vercel), Solomon Hykes (Docker), and Wes McKinney (Pandas creator). Astral's first product, **Ruff**, proved the viability of their approach—a Python linter 10-100x faster than Flake8 that's now used by Airflow, FastAPI, Pandas, and major tech companies.

UV launched on February 15, 2024, initially as a pip and pip-tools replacement. The **August 2024 release** expanded UV into a complete project manager with lockfiles, Python version management, and tool execution. By January 2026, UV had reached version **0.9.26** with 514 contributors and 246 releases. The JetBrains Python Developers Survey 2024 highlighted UV's remarkable 11% adoption rate within its first year—unprecedented for a new packaging tool.

The core value proposition is simple: Python's historically fragmented tooling (pip for packages, virtualenv for environments, pyenv for Python versions, pipx for tools) now consolidates into one binary that requires no Python installation to run.

## Speed that changes developer behavior

UV's performance stems from its **Rust implementation** combined with several technical innovations. The tool achieves **8-10x faster** package installation than pip without caching and **80-115x faster** with a warm cache. Virtual environment creation runs **80x faster** than `python -m venv`.

The speed comes from multiple architectural decisions:

- **Python-free resolution**: UV parses TOML and wheel metadata natively in Rust, only spawning Python for legacy setup.py packages
- **HTTP range requests**: Downloads only wheel metadata (a few kilobytes) rather than entire wheels during resolution, enabled by PEP 658
- **Zero-copy deserialization**: Uses rkyv binary format that matches in-memory representation exactly, allowing O(1) metadata access
- **Parallel operations**: Tokio async runtime for I/O and Rayon thread pool for CPU-intensive tasks
- **Global deduplication cache**: Hard links allow multiple projects to share identical cached files without disk duplication

UV's **PubGrub resolver** (borrowed from Dart's pub package manager) implements **Conflict-Driven Clause Learning (CDCL)** from SAT solvers. When hitting a dependency conflict, it analyzes *why* resolution failed and skips similar dead ends, dramatically outperforming pip's simple backtracking on complex dependency graphs.

## Universal lockfiles and cross-platform resolution

UV introduces **universal lockfiles** (`uv.lock`) that capture resolutions for all platforms regardless of where they're generated. A lockfile created on macOS will correctly install on Linux and Windows because UV's **forking resolver** splits resolution along platform markers:

```toml
[[package]]
name = "requests"
version = "2.31.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "certifi" },
    { name = "charset-normalizer" },
    { name = "idna" },
    { name = "urllib3" },
]
```

The lockfile includes cryptographic hashes for integrity verification and supports export to `requirements.txt`, PEP 751's `pylock.toml`, and CycloneDX SBOM formats. Key lock commands include `uv lock` to generate, `uv sync --locked` to install exactly from lockfile (failing if outdated), and `uv export` for interoperability.

## Installation across all platforms

Install UV via standalone scripts that require no prerequisites:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Homebrew
brew install uv
```

UV installs as a single static binary and self-updates via `uv self update`.

## Essential commands for daily development

**Project initialization** creates a standard structure with `pyproject.toml`:

```bash
uv init my-project      # New project directory
cd my-project
uv add requests pandas  # Add dependencies (creates .venv and uv.lock automatically)
uv add --dev pytest     # Development dependencies
uv run python main.py   # Execute in project environment
uv run pytest           # Run tests
```

**The pip-compatible interface** enables gradual adoption:

```bash
uv pip install requests              # Direct pip replacement
uv pip install -r requirements.txt   # Install from requirements
uv pip compile requirements.in       # Lock dependencies (like pip-tools)
uv pip freeze                        # Export installed packages
```

**Python version management** eliminates the need for pyenv:

```bash
uv python install 3.12 3.13     # Install multiple versions
uv python list                   # Show available versions
uv venv --python 3.12           # Create venv with specific version
uv python pin 3.11              # Pin version in .python-version
```

**Tool execution** with `uvx` (similar to npx or pipx):

```bash
uvx ruff check .        # Run ruff without installing
uvx black --check .     # Run black formatter
uv tool install ruff    # Install tool permanently
```

## Configuration through pyproject.toml

UV embraces PEP 621 for project metadata:

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["requests>=2.28.0", "pandas>=2.0.0"]

[dependency-groups]
dev = ["pytest>=7.0.0", "ruff>=0.1.0"]
test = ["pytest-cov>=4.0.0"]

[tool.uv]
dev-dependencies = ["pytest>=7.0.0"]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cpu"
```

UV also supports **workspaces** for monorepo setups with shared lockfiles across multiple packages.

## UV versus conda: Different tools for different needs

The two tools serve fundamentally different purposes. **UV focuses exclusively on Python** and the PyPI ecosystem, optimizing for speed and modern packaging standards. **Conda is language-agnostic**, managing entire software stacks including C libraries, CUDA toolkits, R packages, and system binaries.

| Aspect | UV | Conda |
|--------|-----|-------|
| Speed | 10-100x faster than pip | Slower, especially resolution |
| Dependencies | PyPI only | Conda-forge + PyPI |
| Non-Python support | None | Excellent (CUDA, MKL, GDAL) |
| BLAS backend | OpenBLAS (bundled) | Swappable (MKL, OpenBLAS) |
| Lock files | Universal cross-platform | Platform-specific |
| Environment model | Project-local (.venv) | Global registry |

**Choose UV for** pure Python projects, web applications (FastAPI, Django), CI/CD pipelines prioritizing speed, and developers wanting modern tooling. **Choose conda for** scientific computing with compiled dependencies, GPU/CUDA workloads needing managed CUDA toolkit, MKL-optimized numerical computing on Intel processors, and cross-language projects mixing R and Python.

## Migrating from conda to UV

Migration works best for projects without heavy non-Python dependencies. Start by exporting your conda environment:

```bash
# Export explicitly installed packages (cleanest)
conda env export --from-history -n myenv > environment.yml

# Get pip packages separately
pip freeze > pip_requirements.txt
```

Initialize the UV project and import dependencies:

```bash
uv init .
uv add -r pip_requirements.txt
uv sync
```

**Handling conda-specific packages** requires case-by-case solutions. Some direct mappings exist: `opencv` becomes `opencv-python`, `pytorch + cudatoolkit` uses PyTorch's wheel index. For non-Python binaries like graphviz or GDAL, use system package managers (`apt install`, `brew install`) or Docker.

For CUDA-dependent projects, the **hybrid approach** often works best:

```bash
# Use conda for CUDA runtime, UV for Python packages
conda create -n cuda-env cudatoolkit cudnn
conda activate cuda-env
uv add torch --index-url https://download.pytorch.org/whl/cu121
```

**Common migration pitfalls** include:

- **Missing MKL optimization**: PyPI NumPy uses OpenBLAS, which performs slightly slower on Intel CPUs—accept this or keep conda for numerical work
- **CUDA management**: Install CUDA toolkit via NVIDIA's installers or system packages, then use PyTorch's wheel index
- **IDE integration**: Point your IDE interpreter to `.venv/bin/python` manually

**Do not migrate** if you rely heavily on conda-managed CUDA/cuDNN, need RAPIDS (which requires conda channels), use many geospatial packages (GDAL ecosystem), or work in regulated environments requiring a single auditable package manager.

## Conclusion

UV represents a generational leap in Python tooling, bringing Rust-powered performance and unified workflows to an ecosystem that historically required mastering multiple tools. For pure Python projects, the migration is straightforward and the speed benefits are immediate—installations that took 20 seconds now complete in under one. The universal lockfile format eliminates "works on my machine" problems for cross-platform teams.

However, UV does not replace conda for scientific computing workflows with complex binary dependencies. The practical approach for many data science teams will be hybrid: conda for CUDA and system libraries, UV for Python package management. As UV continues its rapid development—246 releases in under two years—expect further convergence, but today the choice depends on your project's dependency profile rather than any universal "better" tool.