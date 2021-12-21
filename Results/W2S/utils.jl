module utils
export augment_dataset_8_fold, get_dataset_mean_std, KLD, reparameterize, recon_loss, recon_loss_val, MSE

using Knet,CUDA,Random,Statistics,Base
const F = Float32
Atype = CUDA.functional() ? KnetArray{F} : Array{F};

function augment_dataset_8_fold(xtrn)
    l90 = [rotl90(xtrn[:,:,i],1) for i in 1:size(xtrn)[3]]
    l180 = [rotl90(xtrn[:,:,i],2) for i in 1:size(xtrn)[3]]
    r90 = [rotl90(xtrn[:,:,i],3) for i in 1:size(xtrn)[3]]
    l90 = cat(l90...,dims=3);
    l180 = cat(l180...,dims=3);
    r90 = cat(r90...,dims=3);
    rotated_matrix = cat([xtrn,l90,l180,r90]...,dims=3);
    augmented = cat([rotated_matrix[end:-1:1,:,:],rotated_matrix]...,dims=3)
    return augmented
end

function get_dataset_mean_std(xtrn,xval)
    data = cat(xtrn,xval,dims=3);
    data_mean = mean(data);
    data_std = stdm(data,data_mean);
    return data_mean,data_std
end

function KLD(mu, logvar)
    kl_error = -0.5 * sum(1 .+ (logvar - mu.*mu - exp.(logvar)))
end

function reparameterize(mu,logvar)
    std = exp.(logvar.*0.5)
    epsilon = convert(Atype, randn(F, size(mu)))
    z = mu .+ (epsilon.*std)
    return z 
end

function recon_loss(x, y,noise_params,data_std,sigma_min)
    scale = noise_params[1].*y .+noise_params[2]
    scale = max.(sigma_min/(data_std^2),scale)
    a = -((x.-y).^2)./(2*scale)
    loss_reco = exp.(a)./sqrt.(scale*data_std^2)
    recon_error = - mean(log.(loss_reco.+1e-9))
    return recon_error
end


function MSE(x, y)
    return mean((x - y).^2)
end


end