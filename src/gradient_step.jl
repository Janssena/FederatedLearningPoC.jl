# Load packages:
@info "Loading packages..."
import Random
import Zygote
import JLD2

using Dates
using DeepCompartmentModels

include("lib/data.jl");
include("lib/model.jl");
include("lib/args.jl");

@info "All packages loaded successfully."

# Run main:
PATHS = handle_args(ARGS)
if !(:parameters in keys(PATHS))
    throw(ErrorException("Parameter path not supplied to script. Specify file location via --ps argument: `julia --project=. --ps=/my/parameters.jld2`"))
end

@info "Loading data and model..."
population = load_data(PATHS.data)
dcm = load_model()

ps, st = parse_parameters(PATHS.parameters)
@assert length(population) == length(ps.phi.μ) "Number of data examples do not match with number of random effect parameters"

@info "Calculating gradient..."
rng = Random.Xoshiro()
update_epsilon!(rng, st)
grad = gradient(dcm, population, ps, st)
@info "Gradient calculation completed!"

@info "Saving output to $(PATHS.output)"
JLD2.save(PATHS.output, Dict("grad" => grad))

println("Task ran successfully!")

# Exit:
println("Exiting...")
exit()