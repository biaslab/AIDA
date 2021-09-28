using ReactiveMP
using GraphPPL
using Rocket

# specify flow model
model = FlowModel(2,
    (
        AdditiveCouplingLayer(PlanarFlow()), # defaults to AdditiveCouplingLayer(PlanarFlow(); permute=true)
        AdditiveCouplingLayer(PlanarFlow()),
        AdditiveCouplingLayer(PlanarFlow()),
        AdditiveCouplingLayer(PlanarFlow())
    )
) 

@model function flow_classifier(nr_samples::Int64, params)
    
    # initialize variables
    x_lat  = randomvar(nr_samples)
    y_lat1 = randomvar(nr_samples)
    y_lat2 = randomvar(nr_samples) #where {prod_constraint=ProdPreserveType(NormalWeightedMeanPrecision)}
    y      = datavar(Float64, nr_samples)
    x      = datavar(Vector{Float64}, nr_samples)

    meta  = FlowMeta(compile(model, params))

    # specify observations
    for k = 1:nr_samples

        # specify latent state
        x_lat[k] ~ MvNormalMeanPrecision(x[k], 1e12*diagm(ones(2)))

        # specify transformed latent value
        y_lat1[k] ~ Flow(x_lat[k]) where { meta = meta }
        y_lat2[k] ~ dot(y_lat1[k], [1, 1]) #where {pipeline=LoggerPipelineStage("dot")}

        # specify observations
        y[k] ~ Probit(y_lat2[k]) #where {pipeline=LoggerPipelineStage("probit")} # default: where { pipeline = RequireInbound(in = NormalMeanPrecision(0, 1.0)) }

    end

    # return variables
    return x_lat, x, y_lat1, y_lat2, y

end

function inference_flow_classifier(data_y::Array{Float64,1}, data_x::Array{Array{Float64,1},1}, params)
    
    # fetch number of samples
    nr_samples = length(data_y)

    # define model
    model, (x_lat, x, y_lat1, y_lat2, y) = flow_classifier(nr_samples, params)

    # initialize free energy
    fe_buffer = nothing
    
    # subscribe
    fe_sub = subscribe!(score(BetheFreeEnergy(), model), (fe) -> fe_buffer = fe)

    # update y and x according to observations (i.e. perform inference)
    ReactiveMP.update!(y, data_y)
    ReactiveMP.update!(x, data_x)

    # unsubscribe
    unsubscribe!(fe_sub)
    
    # return the marginal values
    return fe_buffer

end;


@model function flow_planner(m_gains, cov_gains, params)
    
    # initialize variables
    y_goal = datavar(Float64)

    meta  = FlowMeta(compile(model, params))

    x_lat ~ MvNormalMeanCovariance(m_gains, cov_gains) where { q = MeanField() }

    # specify transformed latent value
    y_lat1 ~ Flow(x_lat) where { meta = meta }
    y_lat2 ~ dot(y_lat1, [1, 1])

    # specify observations
    y_goal ~ Probit(y_lat2)

    # return variables
    return x_lat, y_lat1, y_lat2, y_goal

end

function inference_flow_planner(m_gains, cov_gains, goal::Float64, params; vmp_iter=10)
    

    # define model
    model, (x_lat, y_lat1, y_lat2, y_goal) = flow_planner(m_gains, cov_gains, params)

    x_buffer = nothing
    xsub = subscribe!(getmarginal(x_lat), (mx) -> x_buffer = mx)
    
    # initialize free energy
    fe_buffer = keep(Real)
    
    # subscribe
    fe_sub = subscribe!(score(BetheFreeEnergy(), model), fe_buffer)
    
    
    # update y and x according to observations (i.e. perform inference)
    for i in 1:100
        ReactiveMP.update!(y_goal, goal)
    end
    # unsubscribe
    unsubscribe!(fe_sub)
    
    # return the marginal values
    return fe_buffer, x_buffer

end;