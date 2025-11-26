# Get julia
FROM julia:1.11.7

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone github repo
WORKDIR /project
RUN git clone --branch main --single-branch https://github.com/Janssena/FederatedLearningPoC.jl.git . && git checkout 96d52d142d4f0435af11a4ba5c0672d6992b4f51

# Install Julia dependencies
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.precompile(timing = true)'

# Run 
CMD ["sh", "-c", "julia --project=. src/gradient_step.jl --data=${DATA_PATH} --out=${OUTPUT_PATH} --ps=${PARAM_PATH}"]
