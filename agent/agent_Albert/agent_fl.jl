using ForneyLab

# X_train = [[0.5, 0.5], [0.9, 0.3], [0.9, 0.3], [0.9, 0.3], [0.9, 0.3], [0.9, 0.3], [0.9, 0.3], [0.9, 0.3], [0.9, 0.3], [0.9, 0.3], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 1.0], [2.0, 1.0], [2.5, 1.0], [2.5, 1.0], [2.5, 1.0]]
# y_train = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
# n_samples = length(y_train)

# for _ in 1:40-n_samples
#     push!(X_train, rand(X_train))
    
#     if (X_train[end] == [1.0, 0.0]) || (X_train[end] == [2.5, 1.0])
#         push!(y_train, 1.0)
#     else
#         push!(y_train, 0.0)
#     end
# end

# n_samples = length(y_train)

# conn_num = 7
# function dense(w,x)
#     lo1 = 1/(1+exp(-transpose(w[1:2])*x))
#     lo2 = 1/(1+exp(-transpose(w[3:4])*x))
#     1/(1+exp(-transpose(w[5:6])*[lo1, lo2] + w[end]))
# end

# # function dense(w,x)
# #     lo1 = -transpose(w[1:2])*x
# #     lo2 = -transpose(w[3:4])*x
# #     1/(1+exp(-transpose(w[5:6])*[lo1, lo2] + w[end]))
# # end

# # conn_num = 3
# # sigmoid(w,x) = 1/(1+exp(-transpose(w[1:end-1])*x + w[end]))

# g = FactorGraph()

# @RV b ~ GaussianMeanVariance(
#           placeholder(:mu_b, dims=(conn_num,)), 
#           placeholder(:Sigma_b, dims=(conn_num,conn_num)))

# z = Vector{Variable}(undef, n_samples)
# y = Vector{Variable}(undef, n_samples)

# for n in 1:n_samples
#     @eval $(Symbol("func$n"))(w) = dense(w, X_train[$n])
#     # @eval $(Symbol("func$n"))(w) = sigmoid(w, X_train[$n])
#     @RV z[n] ~ Nonlinear{Sampling}(b,g=eval(Symbol("func$n")), n_samples=1000,in_variates=[Multivariate],out_variate=Univariate)
#     @RV y[n] ~ Bernoulli(z[n])
#     placeholder(y[n], :y, index=n)
# end

# algo = messagePassingAlgorithm(b, free_energy=true)
# source_code = algorithmSourceCode(algo, free_energy=true)
# eval(Meta.parse(source_code))

# # Prior statistics
# μ_b = zeros(conn_num)
# Σ_b = diageye(conn_num)

# data = Dict(:mu_b => μ_b,
#             :Sigma_b => Σ_b,
#             :y => y_train)

# marginals = step!(data)

# b_m = ForneyLab.unsafeMean(marginals[:b])

# for n in 1:n_samples
#     p = dense(b_m,X_train[n])
#     # p = sigmoid(b_m,X_train[n])
#     @show p
#     @show y_train[n]
# end


# define inference graph
function build_est_graph(n_gs, nn_params, n_samples=1)
    graph = ForneyLab.FactorGraph()

    @RV gs ~ GaussianMeanVariance(
              placeholder(:mu_gs, dims=(n_gs,)), 
              placeholder(:Sigma_gs, dims=(n_gs,n_gs)))


    z = Vector{Variable}(undef, n_samples)
    y = Vector{Variable}(undef, n_samples)
    for n in 1:n_samples
        @eval $(Symbol("func$n"))(x) = nn_forward(x, nn_params)
        @RV z[n] ~ Nonlinear{Sampling}(gs,g=eval(Symbol("func$n")),in_variates=[Multivariate],out_variate=Univariate)
        if n == n_samples
            @RV y[n] ~ GaussianMeanVariance(z[n], 1e-4)
            placeholder(y[n], :y, index=n)
        else
            @RV y[n] ~ Bernoulli(z[n])
            placeholder(y[n], :y, index=n)
        end
    end
    
    return graph
end