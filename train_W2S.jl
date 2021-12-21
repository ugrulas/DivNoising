module W2S
using Knet, CUDA, MLDatasets, ArgParse, Images, Random, Statistics,Base,NPZ
using Memento
const F = Float32
Atype = CUDA.functional() ? KnetArray{F} : Array{F};

include("utils.jl")
using .utils


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

    push!(θ, xavier(3, 3, channel*2, channel*2))
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
        
    
    push!(ϕ, xavier(3, 3, channel*2, channel*2))
    push!(ϕ, zeros(1, 1, channel*2, 1))
    
    push!(ϕ, xavier(3, 3, channel*2, channel*2))
    push!(ϕ, zeros(1, 1, channel*2, 1))

    ϕ = map(a->convert(Atype,a), ϕ)
    noise_params = []
    push!(noise_params,ones(1))
    push!(noise_params,zeros(1))
    noise_params = map(a->convert(Atype,a), noise_params)
    return θ,ϕ,noise_params
end


function recon_loss_FU(x, y,noise_params,data_std,sigma_min)
    scale = noise_params[1].*y .+noise_params[2]
    scale = max.(sigma_min/(data_std^2),scale)
    a = -((x.-y).^2)./(2*scale)
    loss_reco = exp.(a)./sqrt.(2.0*3.14*scale)
    recon_error = - mean(log.(loss_reco.+1e-9))
    return recon_error
end

function loss(w,x,data_mean,data_std,epoch,min_var)
    wdec, wenc,noise_params = w[:decoder], w[:encoder],w[:noise_params]
    x_normalized = (x .- data_mean)./ data_std
    mu, logvar = encode(wenc, x_normalized)
    z = reparameterize(mu,logvar)
    y = decode(wdec,z)
    return recon_loss_FU(x_normalized,y,noise_params,data_std,min_var) + (KLD(mu,logvar)/(128*128*32))
end
lossgradient = grad(loss);

function loss_val(w, x,data_mean,data_std,min_var)
    wdec, wenc,noise_params = w[:decoder], w[:encoder],w[:noise_params]
    x_normalized = (x .- data_mean)./ data_std
    mu, logvar = encode(wenc, x_normalized)
    z = reparameterize(mu,logvar)
    y = decode(wdec,z)
    return recon_loss_FU(x_normalized,y,noise_params,data_std,min_var), KLD(mu,logvar)/(128*128*32)
end

function main(args="")
    s = ArgParseSettings()
    s.description="Fully Unsupervised Training on W2S datasets."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=32; help="minibatch size")
        ("--epochs"; arg_type=Int; default=1000; help="number of epochs for training")
        ("--nf"; arg_type=Int; default=32; help="number of filters for first conv layer of encoder")
        ("--lr"; arg_type=Float64; default=1e-3; help="initial learning rate") 
        ("--channel"; arg_type=Int; help = "W2S dataset channel to train 0,1 or 2")
        ("--avg"; arg_type=Int; help = "W2S 1 or 16 corresponds to Avg1 and Avg 16 respectively")
        ("--minvar"; arg_type=Float64; help="minimum allowed variance [original implementation uses 1.0 for Avg16 9.0 for Avg1]") 
    end
    
    isa(args, String) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && Knet.seed!(o[:seed])
    channel = o[:channel]
    avg = o[:avg]
    experiment = string("w2s_avg",avg, "_channel_",channel)
    train_data = npzread("../Datasets/"*experiment*".npz");
    xtrn=train_data["train"];
    xval=train_data["val"];
    xtrn=permutedims(xtrn, [2, 3, 1]);
    xval=permutedims(xval, [2, 3, 1]);
    batch_size = o[:batchsize]
    augmented = augment_dataset_8_fold(xtrn)
    dtrn = minibatch(augmented, batch_size; xsize = (128, 128, 1,:), xtype = Atype,shuffle=true)
    dval = minibatch(xval, batch_size; xsize = (128, 128, 1,:), xtype = Atype);
    data_mean,data_std = get_dataset_mean_std(xtrn,xval)
    min_var = o[:minvar]
    wdec, wenc,noise_params = weights(o[:nf]);
    initial_lr = o[:lr]
    w = Dict(
           :encoder => wenc,
           :decoder => wdec,
           :noise_params => noise_params,
           )

    opt = Dict(
            :encoder => map(wi->Knet.Adam(lr = initial_lr), w[:encoder]),
            :decoder => map(wi->Knet.Adam(lr = initial_lr), w[:decoder]),
            :noise_params => map(wi->Knet.Adam(lr = initial_lr), w[:noise_params]),
        )

    logger = Memento.config!("info"; fmt="[{date} | {level} | {name}]: {msg}");
    push!(logger, DefaultHandler(experiment*".log",DefaultFormatter("[{date} | {level} | {name}]: {msg}")));
    
    trn_recon_loss = Float64[]
    val_recon_loss = Float64[]
    trn_KLD_loss = Float64[]
    val_KLD_loss = Float64[]

    number_of_epochs = o[:epochs]
    best_val_loss = 10^10
    best_val_loss_epoch = 0
    val_len = length(dval)
    trn_len = length(dtrn)

    for epoch = 1:number_of_epochs
        for (i,x)  in enumerate(dtrn)
            dw = lossgradient(w, x,data_mean,data_std,epoch,min_var)
            update!(w, dw, opt)
        end

        mse_sum = 0
        kld_sum = 0
        for (j,x) in enumerate(dtrn)
            mse,kld  = loss_val(w,x,data_mean,data_std,min_var) 
            mse_sum += mse
            kld_sum += kld
        end
        info(logger,"Epoch : $epoch")
        mse_sum = mse_sum/trn_len
        kld_sum = kld_sum/trn_len
        info(logger,"Train set -> recon_loss : $mse_sum, KLD : $kld_sum")
        push!(trn_recon_loss, mse_sum)
        push!(trn_KLD_loss, kld_sum)


        mse_sum = 0
        kld_sum = 0
        for (j,x) in enumerate(dval)
            mse,kld  = loss_val(w,x,data_mean,data_std,min_var) 
            mse_sum += mse
            kld_sum += kld
        end

        mse_sum = mse_sum/val_len
        kld_sum = kld_sum/val_len
        push!(val_recon_loss, mse_sum)
        push!(val_KLD_loss, kld_sum)
        total_loss = mse_sum + kld_sum

        info(logger,"Validation set -> recon_loss : $mse_sum, KLD : $kld_sum")
        if total_loss<best_val_loss
            best_val_loss = total_loss
            best_val_loss_epoch = epoch
            info(logger,"!!New Best!!")
            Knet.save(experiment*"_best.jld2","w",w)
        end

        if (epoch - best_val_loss_epoch)>30
            for layer in opt[:encoder]
                layer.lr *= 0.5
            end
            for layer in opt[:decoder]
                layer.lr *= 0.5
            end 
            for layer in opt[:noise_params]
                layer.lr *= 0.5
            end 
            info(logger,"Reduced lr")
            best_val_loss_epoch = epoch
        end
        Knet.save(experiment*"_last.jld2","w",w)
        info(logger,"=======================================\n")
    end

    Knet.save(experiment*"_trn_recon_loss_fully_unsupervised.jld2","trn_recon_loss",trn_recon_loss) 
    Knet.save(experiment*"_trn_KLD_loss_fully_unsupervised.jld2","trn_KLD_loss",trn_KLD_loss) 
    Knet.save(experiment*"_val_recon_loss_fully_unsupervised.jld2","val_recon_loss",val_recon_loss) 
    Knet.save(experiment*"_val_KLD_loss_fully_unsupervised.jld2","val_KLD_loss",val_KLD_loss) 
end

PROGRAM_FILE == "train_W2S.jl" && main(ARGS)

end # module