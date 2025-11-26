# PoC Haemophilia model

This repo contains the code for running a simple local job on a federated network.

### Local optimisation of random effect parameters

The Julia code for the model to find MAP estimates of random effect posteriors can be run using the following command:

```bash
julia --project=. src/local_optimisation.jl --data=DATA_PATH --out=OUTPUT_PATH
```

Here, the user is expected to supply two arguments:

`DATA_PATH`: the path pointing to a .csv file containing the data. Dummy data is available under the `/data` folder.

`OUTPUT_PATH`: the path where the resulting .json file is saved.

Each node should have its own local data set. This specific model performs a local job, meaning that the result is specific
to that node. The central node thus only needs to aggregate the data by concatenating the results from each node. Alternatively,
the central node stores the result from each node in a object pointing to that specific node.

An example in pseudo-code:

```julia
results = map(nodes) do node
    run_local_job(data_path[node], output_path[node]) # returns the result.json
end

concat(results)
# or 
result = NamedTuple(node_1 = results[1], node_2 = results[2], ) # etc.
```

### Gradient calculation of global parameters

The Julia code to calculate the gradients with respect to model global and local parameters can be run using the following command:

```bash
julia --project=. src/gradient_step.jl --data=DATA_PATH --out=OUTPUT_PATH --ps=PARAM_PATH
```

Here, the user is expected to supply three arguments:

`DATA_PATH`: the path pointing to a .csv file containing the data. Dummy data is available under the `/data` folder.

`OUTPUT_PATH`: the path where the resulting .jld2 file is saved. 

`PARAM_PATH`: the path where the current parameter object file is saved. Initial parameters are provided in the `/data` folder.

The docker implementation can be build and ran as follows:

```bash
$ docker build -t global-update docker/global/.
$ docker run \
    -e DATA_PATH="data/simulated_data_centre_1.csv" \
    -e OUTPUT_PATH="checkpoints/gradient_centre_1.jld2" \
    -e PARAM_PATH="data/parameters_centre_1.jld2" \
    global-update
```