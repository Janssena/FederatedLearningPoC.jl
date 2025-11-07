# Get julia
FROM julia:1.11.7

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone github repo
WORKDIR /project
RUN git clone --branch main --single-branch https://github.com/Janssena/FederatedLearningPoC.jl.git .
RUN git checkout 6156ffda071f7052079578a408b011c031c2dbdd

# Install Julia dependencies
RUN julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); Pkg.precompile(timing = true)'

# Run 
CMD ["sh", "-c", "julia --project=. src/main.jl --data ${DATA_PATH} --ckpt ${CKPT_PATH}"]
