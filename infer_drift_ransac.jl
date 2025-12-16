using GLMakie, GeometryBasics

#######################################
##   Render Wireframe w/ GLMakie     ##
#######################################

function render_wireframe_makie(vertices::AbstractMatrix,
                                edges::Vector{<:Tuple{Int,Int}};
                                width::Int=256, height::Int=256,
                                azimuth::Real=pi, elevation::Real=pi/6,
                                linewidth::Real=2,
                                linecolor=:black,
                                bgcolor=:white)
    fig = Figure(size=(width, height), backgroundcolor=bgcolor)
    ax  = Axis3(fig[1,1]; aspect=:data, perspectiveness=0.9, backgroundcolor=bgcolor)
    hidedecorations!(ax); hidespines!(ax)
    pts  = Point3f.(eachrow(vertices))
    segs = [pts[i] => pts[j] for (i,j) in edges]
    linesegments!(ax, segs; linewidth, color=linecolor)
    ax.azimuth[] = azimuth
    ax.elevation[] = elevation
    img = colorbuffer(fig.scene)
    return img, fig
end

V = Float32.([0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1])
E = [(1,2),(2,3),(3,4),(4,1),(5,6),(6,7),(7,8),(8,5),(1,5),(2,6),(3,7),(4,8)]

az = pi
el = 0.5*pi
obs_img, obs_fig = render_wireframe_makie(V, E; width=256, height=256, azimuth=az, elevation=el);
println("True azimuth: ", az);
println("True elevation: ", el);
save("obs_img.png", obs_img);

#######################################
##         RANSAC + Drift MH         ##
#######################################
using Gen, ImageCore, Plots
using Statistics
using LinearAlgebra
using Images

_to_chw3(a) = begin
    if a isa AbstractMatrix{<:Colorant}
        Float32.(channelview(a)[1:3, :, :])
    elseif a isa AbstractArray{<:Real,3}
        A = Float32.(a)
        size(A,1) == 4 ? @view(A[1:3, :, :]) : A
    else
        Float32.(channelview(a)[1:3, :, :])
    end
end

function img_to_gray_array(img)
    Float32.(channelview(Gray.(img)))
end

function edge_mask(img; thresh::Real=0.5)
    gray = img_to_gray_array(img)
    BitMatrix(gray .< thresh)
end

function edge_coords(edges::BitMatrix)
    idxs = findall(edges)
    n = length(idxs)
    xs = Array{Float64}(undef, n)
    ys = Array{Float64}(undef, n)
    for (i, idx) in enumerate(idxs)
        ys[i] = idx[1]
        xs[i] = idx[2]
    end
    hcat(xs, ys)
end

function normalize_points(pts::AbstractMatrix{<:Real})
    if size(pts, 1) == 0
        return Array{Float64}(undef, 0, 2)
    end
    ptsf = Array{Float64}(pts)
    μ = vec(mean(ptsf, dims=1))
    pts0 = ptsf .- μ'
    r = maximum(sqrt.(sum(abs2, pts0; dims=2)))
    if r == 0.0
        return pts0
    end
    pts0 ./ r
end

function chamfer_distance(obs_edges::BitMatrix, pred_edges::BitMatrix)
    As_raw = edge_coords(obs_edges)
    Bs_raw = edge_coords(pred_edges)

    if size(As_raw, 1) == 0 && size(Bs_raw, 1) == 0
        return 0.0
    elseif size(As_raw, 1) == 0 || size(Bs_raw, 1) == 0
        return Inf
    end

    As = normalize_points(As_raw)
    Bs = normalize_points(Bs_raw)

    dA = 0.0
    for i in 1:size(As, 1)
        ax = As[i, 1]
        ay = As[i, 2]
        min_d2 = Inf
        for j in 1:size(Bs, 1)
            bx = Bs[j, 1]
            by = Bs[j, 2]
            dx = ax - bx
            dy = ay - by
            d2 = dx*dx + dy*dy
            if d2 < min_d2
                min_d2 = d2
            end
        end
        dA += sqrt(min_d2)
    end
    dA /= size(As, 1)

    dB = 0.0
    for j in 1:size(Bs, 1)
        bx = Bs[j, 1]
        by = Bs[j, 2]
        min_d2 = Inf
        for i in 1:size(As, 1)
            ax = As[i, 1]
            ay = As[i, 2]
            dx = bx - ax
            dy = by - ay
            d2 = dx*dx + dy*dy
            if d2 < min_d2
                min_d2 = d2
            end
        end
        dB += sqrt(min_d2)
    end
    dB /= size(Bs, 1)

    (dA + dB) / 2
end

struct ImageChamfer <: Gen.Distribution{Any} end
const image_chamfer = ImageChamfer()

Gen.logpdf(::ImageChamfer, y, renderer::Function, sig::Real) = begin
    obs_edges = edge_mask(y)
    pred_edges = edge_mask(renderer())
    D = chamfer_distance(obs_edges, pred_edges)
    isfinite(D) ? -(D^2) / (2 * sig) : -Inf
end

Gen.random(::ImageChamfer, renderer::Function, sig::Real) = renderer()
Gen.has_output_grad(::ImageChamfer) = false
Gen.has_argument_grads(::ImageChamfer) = (false, false)
Gen.is_discrete(::ImageChamfer) = false

@gen function wireframe_camera_model(vertices, edges, width::Int, height::Int, sig::Real)
    az ~ uniform_continuous(0, 2*pi)
    el ~ uniform_continuous(-pi/2, pi/2)

    renderer = () -> begin
        img, _ = render_wireframe_makie(vertices, edges; width=width, height=height,
                                        azimuth=az, elevation=el)
        img
    end

    {:img} ~ image_chamfer(renderer, sig)
    return (az, el)
end

function view_loglik(vertices, edges, obs_img;
                     width::Int=256, height::Int=256, sig::Real=0.05,
                     az::Real=0.0, el::Real=0.0)
    renderer = () -> begin
        img, _ = render_wireframe_makie(vertices, edges;
                                        width=width, height=height,
                                        azimuth=az, elevation=el)
        img
    end
    Gen.logpdf(image_chamfer, obs_img, renderer, sig)
end

function camera_ransac(vertices, edges, obs_img;
                       width::Int=256, height::Int=256, sig::Real=0.05,
                       num_candidates::Int=200)
    best_ll = -Inf
    best_az = 0.0
    best_el = 0.0

    for _ in 1:num_candidates
        az = 2pi * rand()
        el = -pi/2 + pi * rand()
        ll = view_loglik(vertices, edges, obs_img;
                         width=width, height=height, sig=sig,
                         az=az, el=el)
        if ll > best_ll
            best_ll, best_az, best_el = ll, az, el
        end
    end
    best_az, best_el
end

const SIG_AZ = 0.05
const SIG_EL = 0.05

@gen function camera_drift_proposal(prev_trace)
    az_prev = prev_trace[:az]
    el_prev = prev_trace[:el]
    az ~ normal(az_prev, SIG_AZ)
    el ~ normal(el_prev, SIG_EL)
end

function gaussian_drift_update(tr)
    (tr, _) = mh(tr, camera_drift_proposal, ())
    tr
end

@gen function ransac_proposal(prev_trace, vertices, edges, obs_img, width::Int, height::Int, sig::Real)
    az_guess, el_guess = camera_ransac(vertices, edges, obs_img;
                                       width=width, height=height, sig=sig)
    az ~ normal(az_guess, 0.1)
    el ~ normal(el_guess, 0.1)
end

function ransac_update(tr, vertices, edges, obs_img;
                       width::Int=256, height::Int=256, sig::Real=0.05)
    (tr, _) = mh(tr, ransac_proposal, (vertices, edges, obs_img, width, height, sig))
    for _ in 1:20
        tr = gaussian_drift_update(tr)
    end
    tr
end

function gaussian_drift_inference(vertices, edges, obs_img;
                                  width::Int=256, height::Int=256, sig::Real=0.05,
                                  steps::Int=1000)
    cons = Gen.choicemap()
    cons[:img] = obs_img
    (tr, _) = Gen.generate(wireframe_camera_model, (vertices, edges, width, height, sig), cons)
    for _ in 1:steps
        tr = gaussian_drift_update(tr)
    end
    tr
end

function ransac_inference(vertices, edges, obs_img;
                          width::Int=256, height::Int=256, sig::Real=0.05,
                          steps::Int=200)
    cons = Gen.choicemap()
    cons[:img] = obs_img
    (tr, _) = Gen.generate(wireframe_camera_model, (vertices, edges, width, height, sig), cons)
    tr = ransac_update(tr, vertices, edges, obs_img; width=width, height=height, sig=sig)
    for _ in 1:steps
        tr = gaussian_drift_update(tr)
    end
    tr
end

visualize_trace(tr; title="") = begin
    az = tr[:az]
    el = tr[:el]
    img, _ = render_wireframe_makie(V, E; width=256, height=256, azimuth=az, elevation=el)
    Plots.plot(img, axis=false, border=false, title=title)
end

function animate_drift(tr; steps=500, path="vis.gif", fps=20)
    anim = Plots.@animate for i in 1:steps
        tr = gaussian_drift_update(tr)
        visualize_trace(tr; title="Iteration $i/$steps")
    end
    Plots.gif(anim, path, fps=fps)
    tr
end

tr = ransac_inference(V, E, obs_img; width=256, height=256, sig=0.05, steps=500);
println("Estimated azimuth:", tr[:az]);
println("Estimated elevation:", tr[:el]);
animate_drift(tr; steps=500, path="vis.gif", fps=20);

println("Best pose log-likelihood: ",
        view_loglik(V, E, obs_img; width=256, height=256, sig=0.05, az=tr[:az], el=tr[:el]));
