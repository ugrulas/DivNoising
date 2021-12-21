module model
export encode, decode, weights

using Knet, CUDA, Random,Statistics,Base
const F = Float32
Atype = CUDA.functional() ? KnetArray{F} : Array{F}

function encode(ϕ, x)

    x = conv4(ϕ[1], x, padding=1)
    x = relu.(x .+ ϕ[2])
    x = conv4(ϕ[3], x, padding=1, stride=1)
    x = relu.(x .+ ϕ[4])
    x = pool(x, stride = 2, mode=0)

    x = conv4(ϕ[5], x, padding=1)
    x = relu.(x .+ ϕ[6])
    x = conv4(ϕ[7], x, padding=1, stride=1)
    x = relu.(x .+ ϕ[8])
    x = pool(x, stride = 2, mode=0)
   
    mu = conv4(ϕ[9], x, padding=1, stride=1)
    mu = mu .+ ϕ[10]
    
    logvar = conv4(ϕ[11], x, padding=1, stride=1)
    logvar = logvar .+ ϕ[12]
    
    return mu, logvar
end

function decode(θ, z)

    z = conv4(θ[1], z, padding=1)
    z = relu.(z .+ θ[2])
    z = conv4(θ[3], z, padding=1, stride=1)
    z = relu.(z .+ θ[4])
    z = deconv4(θ[5], z, padding=0, stride=2)
    z = relu.(z .+ θ[6])


    z = conv4(θ[7], z, padding=1)
    z = relu.(z .+ θ[8])
    z = conv4(θ[9], z, padding=1, stride=1)
    z = relu.(z .+ θ[10]) 
    z = deconv4(θ[11], z, padding=0, stride=2)
    z = relu.(z .+ θ[12])

    z = conv4(θ[13], z, padding=1)
    z = z .+ θ[14]

    return z
end

function weights(channel)
    
    θ = [] # z->x

    push!(θ, xavier(3, 3, 8, channel*2))
    push!(θ, zeros(1, 1, channel*2, 1))

    push!(θ, xavier(3,3,channel*2,channel*2))
    push!(θ, zeros(1, 1, channel*2, 1))
    
    push!(θ, xavier(2, 2, channel*2, channel*2))
    push!(θ, zeros(1,1,channel*2,1))
    
    
    push!(θ, xavier(3, 3, channel*2, channel))
    push!(θ, zeros(1,1, channel,1))
    
    push!(θ, xavier(3,3,channel,channel))
    push!(θ, zeros(1, 1, channel, 1))
    
    push!(θ, xavier(2,2,channel,channel))
    push!(θ, zeros(1, 1, channel, 1))
    

    push!(θ, xavier(3,3,channel,1))
    push!(θ, zeros(1, 1, 1, 1))
    
    
    θ = map(a->convert(Atype,a), θ)

    ϕ = [] # x->z

    push!(ϕ, xavier(3, 3, 1, channel))
    push!(ϕ, zeros(1, 1, channel, 1))

    push!(ϕ, xavier(3, 3, channel, channel))
    push!(ϕ, zeros(1, 1, channel, 1))
    
    
    push!(ϕ, xavier(3, 3, channel, channel*2))
    push!(ϕ, zeros(1,1, channel*2, 1))

    push!(ϕ, xavier(3, 3, channel*2,channel*2))
    push!(ϕ, zeros(1, 1, channel*2, 1))
        
    
    push!(ϕ, xavier(3, 3, channel*2, 8))
    push!(ϕ, zeros(1, 1, 8, 1))
    
    push!(ϕ, xavier(3, 3, channel*2, 8))
    push!(ϕ, zeros(1, 1, 8, 1))

    ϕ = map(a->convert(Atype,a), ϕ)
    noise_params = []
    push!(noise_params,ones(1))
    push!(noise_params,zeros(1))
    noise_params = map(a->convert(Atype,a), noise_params)
    return θ,ϕ,noise_params
end
end