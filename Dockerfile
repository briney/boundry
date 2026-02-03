# Boundry Docker Image
# Combines LigandMPNN sequence design with OpenMM AMBER relaxation
#
# Build: docker build -t boundry .
# Run:   docker run --rm -v $(pwd):/data boundry -i /data/input.pdb -o /data/output.pdb

FROM condaforge/miniforge3:latest

LABEL org.opencontainers.image.source="https://github.com/bryanbriney/boundry"
LABEL org.opencontainers.image.description="Boundry: LigandMPNN + OpenMM AMBER relaxation"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Accept version as build argument for setuptools-scm
ARG VERSION=0.0.0.dev0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BOUNDRY=${VERSION}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files 
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY scripts/ ./scripts/

# Install conda dependencies and package
# Note: openmm and pdbfixer are only available on conda-forge, not PyPI
RUN mamba install -y -c conda-forge python=3.11 openmm pdbfixer pytorch-cpu prody \
    && mamba clean -afy \
    && pip install --no-cache-dir -e .

# Download model weights (~40MB total) to a shared location
ENV BOUNDRY_WEIGHTS_DIR=/app/weights
RUN ./scripts/download_weights.sh

# Create non-root user for security
RUN useradd -m boundry && chown -R boundry:boundry /app
USER boundry

ENTRYPOINT ["boundry"]
CMD ["--help"]
