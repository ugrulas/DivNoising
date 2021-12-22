module DenoiSegMouse
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



function recon_loss_Pnm(x, y,gaus_std,data_std)
    return mean((x - y).^2) / (2.0*(gaus_std/data_std)^2)
end


function recon_loss_FU(x, y,noise_params,data_std,min_var)
    scale = noise_params[1].*y .+noise_params[2]
    scale = max.(min_var/(data_std^2),scale)
    a = -((x.-y).^2)./(2*scale)
    loss_reco = exp.(a)./sqrt.(2.0*3.14*scale)
    recon_error = - mean(log.(loss_reco.+1e-9))
    return recon_error
end

function loss(w,x,data_mean,data_std,epoch,min_var,gauss_std,kl_anneal_epochs,batch_size,trn_recon_epoch,trn_kld_epoch)
    wdec, wenc,noise_params = w[:decoder], w[:encoder],w[:noise_params]
    x_normalized = (x .- data_mean)./ data_std
    mu, logvar = encode(wenc, x_normalized)
    z = reparameterize(mu,logvar)
    y = decode(wdec,z)
    if gauss_std < 0
        recon_loss = recon_loss_FU(x_normalized,y,noise_params,data_std,min_var)
        kld_loss = (KLD(mu,logvar)/(128*128*batch_size))
        push!(trn_recon_epoch,recon_loss)
        push!(trn_kld_epoch,kld_loss)
        return recon_loss + kld_loss
    else
        if kl_anneal_epochs>0
            kl_weight = min(1.0,(epoch-1)/kl_anneal_epochs) 
        else
            kl_weight = 1.0
        end
        recon_loss = recon_loss_Pnm(x_normalized,y,gauss_std,data_std)
        kld_loss =  (KLD(mu,logvar)/(128*128*batch_size))
       
        push!(trn_recon_epoch,value(recon_loss))
        push!(trn_kld_epoch,value(kld_loss))
        return recon_loss + (kl_weight *kld_loss)
    end
end

lossgradient = grad(loss);

function loss_val(w, x,data_mean,data_std,epoch,min_var,gauss_std,kl_anneal_epochs,batch_size)
    wdec, wenc,noise_params = w[:decoder], w[:encoder],w[:noise_params]
    x_normalized = (x .- data_mean)./ data_std
    mu, logvar = encode(wenc, x_normalized)
    z = reparameterize(mu,logvar)
    y = decode(wdec,z)
    if gauss_std < 0
        return recon_loss_FU(x_normalized,y,noise_params,data_std,min_var) , (KLD(mu,logvar)/(128*128*batch_size))
    else
        if kl_anneal_epochs>0
            kl_weight = min(1.0,(epoch-1)/kl_anneal_epochs) 
        else
            kl_weight = 1.0
        end
        return recon_loss_Pnm(x_normalized,y,gauss_std,data_std) , (KLD(mu,logvar)/(128*128*batch_size))
    end
end



function main(args="")
    s = ArgParseSettings()
    s.description="Fully Unsupervised  or Unsupervised Training on DenoiSeg Mouse and DenoiSeg Mouse s&p datasets."
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=32; help="minibatch size")
        ("--epochs"; arg_type=Int; default=1000; help="number of epochs for training")
        ("--nf"; arg_type=Int; default=32; help="number of filters for first conv layer of encoder")
        ("--lr"; arg_type=Float64; default=1e-3; help="initial learning rate") 
        ("--gauss_std"; arg_type=Float64;default=-1.0; help = "Gaussian noise assumption for the dataset,if it is negative fully unsupervised traning is performed")
        ("--kl_anneal"; arg_type=Int; default = 0;help ="How many epochs to perform kl annealing starting from first epoch. [To prevent posterior collapse, KL annealing aproach might be needed.]")
        ("--minvar"; arg_type=Float64;default=-1.0; help="minimum allowed variance required for [original implementation uses 9.0 for Mouse and  1.0 for Mouse s&p]") 
        ("--dataset" ; arg_type=Int;default=0; help="Choose 0 for  Mouse and 1 for Mouse s&p") 
    end
    
    isa(args, String) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    o = parse_args(args, s; as_symbols=true)
    o[:seed] > 0 && Knet.seed!(o[:seed])
    number_of_epochs = o[:epochs]
    dataset = o[:dataset]
    gauss_std = o[:gauss_std]
    kl_anneal_epochs = o[:kl_anneal]
    min_var = o[:minvar]
    initial_lr = o[:lr]
    batch_size = o[:batchsize]
    
    if gauss_std<0 && min_var<0
        return ArgumentError("Either gauss_std or min_var must be provided")
    end
    
    if dataset == 0
        experiment = "DenoiSeg_Mouse"
        train_data = npzread("../Datasets/DenoiseSeg_Mouse_20.npz");
    elseif dataset ==1
        experiment = "DenoiSeg_Mouse_s&p"
        train_data = npzread("../Datasets/DenoiseSeg_Mouse_s&p.npz");
    end
    xtrn=train_data["train"];
    xval=train_data["val"];
    xtrn=permutedims(xtrn, [2, 3, 1]);
    xval=permutedims(xval, [2, 3, 1]);

    augmented = augment_dataset_8_fold(xtrn)
    dtrn = minibatch(augmented, batch_size; xsize = (128, 128, 1,:), xtype = Atype,shuffle=true)
    dval = minibatch(xval, batch_size; xsize = (128, 128, 1,:), xtype = Atype);
    data_mean,data_std = get_dataset_mean_std(xtrn,xval)
    wdec, wenc,noise_params = weights(o[:nf]);

    
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

    
    best_val_loss = 10^10
    best_val_loss_epoch = 0
    val_len = length(dval)
    trn_len = length(dtrn)
    
    for epoch = 1:number_of_epochs
        info(logger,"Epoch : $epoch")
        trn_recon_epoch = []
        trn_kld_epoch = []
        for (i,x)  in enumerate(dtrn)
            dw =lossgradient(w,x,data_mean,data_std,epoch,min_var,gauss_std,kl_anneal_epochs,batch_size,trn_recon_epoch,trn_kld_epoch)
            update!(w, dw, opt)
        end

       
        mse_sum = mean(trn_recon_epoch)
        kld_sum = mean(trn_kld_epoch)
        info(logger,"Train set -> recon_loss : $mse_sum, KLD : $kld_sum")
        push!(trn_recon_loss, mse_sum)
        push!(trn_KLD_loss, kld_sum)


        mse_sum = 0
        kld_sum = 0
        for (j,x) in enumerate(dval)
            mse,kld  = loss_val(w,x,data_mean,data_std,epoch,min_var,gauss_std,kl_anneal_epochs,batch_size) 
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

PROGRAM_FILE == "train_DenoiSeg_Mouse.jl" && main(ARGS)

end # module