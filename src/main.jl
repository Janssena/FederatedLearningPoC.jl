# Download & precompile packages:

@info "Instantiating project..."

import Pkg

Pkg.instantiate(verbose = true)

# Load packages:

@info "Loading packages..."

import Random
import JSON

using Dates
using DeepCompartmentModels

include("lib/data.jl");
include("lib/optimise.jl");
include("lib/helpers.jl");

@info "All packages loaded successfully."

# Run main:

PATHS = handle_args(ARGS)

rng = Random.Xoshiro(42)

@info "Loading data..."
population = load_data(PATHS.data)

@info "Starting optimistation..."
ps_opt = run_optimisation(population)
@info "Optimisation completed!"

@info "Saving checkpoint to $(PATHS.ckpt)"
JSON.json(PATHS.ckpt, ps_opt, pretty = true)

println("Task ran successfully!")

# Exit:

println("Exiting...")
exit()