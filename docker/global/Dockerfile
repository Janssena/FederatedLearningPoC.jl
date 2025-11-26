# Get julia
FROM julia:1.11.7

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone github repo
WORKDIR /project
RUN git clone --branch main --single-branch https://github.com/Janssena/FederatedLearningPoC.jl.git .
RUN git checkout 49318a358c689d9a3021adbc94abbbe0913a10ef

# Install Julia dependencies
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.precompile(timing = true); include("src/precompile_gradient.jl")'

# Run 
CMD ["sh", "-c", "julia --project=. src/gradient_step.jl --data ${DATA_PATH} --out ${OUT_PATH} --ps ${PARAM_PATH}"]
