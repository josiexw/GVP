using GLMakie, GeometryBasics

#######################################
##   Render Wireframe w/ GLMakie     ##
#######################################

function render_wireframe_makie(vertices::AbstractMatrix,
                                edges::Vector{<:Tuple{Int,Int}};
                                width::Int=256, height::Int=256,
                                azimuth::Real=pi, elevation::Real=pi/6,
                                perspectiveness::Real=0.9,
                                linewidth::Real=2,
                                linecolor=:black,
                                bgcolor=:white)
    fig = Figure(size=(width, height), backgroundcolor=bgcolor)
    ax  = Axis3(fig[1,1]; aspect=:data, perspectiveness=perspectiveness, backgroundcolor=bgcolor)
    hidedecorations!(ax); hidespines!(ax)
    pts  = Point3f.(eachrow(vertices))
    segs = [pts[i] => pts[j] for (i,j) in edges]
    linesegments!(ax, segs; linewidth, color=linecolor)
    ax.azimuth[] = azimuth
    ax.elevation[] = elevation
    autolimits!(ax)
    img = colorbuffer(fig.scene)
    return img, fig
end

V = Float32.([0.0 0.0 0.0; 1.0 0.0 0.0; 1.0 0.3 0.0; 0.3 0.3 0.0; 0.3 1.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 1.0; 1.0 0.3 1.0; 0.3 0.3 1.0; 0.3 1.0 1.0; 0.0 1.0 1.0])
E = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 7), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]

az = 6.28
el = 0.001
obs_img, obs_fig = render_wireframe_makie(V, E; width=256, height=256, azimuth=az, elevation=el, perspectiveness=0.9);
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

global_best_ll = Ref(-Inf)

Gen.logpdf(::ImageChamfer, y, vertices, edges, width::Int, height::Int, az::Real, el::Real, sig::Real) = begin
    obs_edges = edge_mask(y)
    pred_img, _ = render_wireframe_makie(vertices, edges; width=width, height=height,
                                         azimuth=az, elevation=el, perspectiveness=0.9)
    pred_edges = edge_mask(pred_img)
    D = chamfer_distance(obs_edges, pred_edges)
    ll = -(D^2) / (2 * sig)
    if ll > global_best_ll[]
        global_best_ll[] = ll
    end
    ll
end

Gen.random(::ImageChamfer, renderer::Function, sig::Real) = renderer()
Gen.has_output_grad(::ImageChamfer) = false
Gen.has_argument_grads(::ImageChamfer) = (false, false)
Gen.is_discrete(::ImageChamfer) = false

@gen function wireframe_camera_model(vertices, edges, width::Int, height::Int, sig::Real)
    az ~ uniform_continuous(0, 2*pi)
    el ~ uniform_continuous(-pi/2, pi/2)
    {:img} ~ image_chamfer(vertices, edges, width, height, az, el, sig)
    return (az, el)
end

function camera_ransac(vertices, edges, obs_img;
                       width::Int=256, height::Int=256, sig::Real=0.001,
                       num_candidates::Int=200)
    best_ll = -Inf
    best_az = 0.0
    best_el = 0.0

    for _ in 1:num_candidates
        az = 2pi * rand()
        el = -pi/2 + pi * rand()
        ll = Gen.logpdf(image_chamfer, obs_img, vertices, edges, width, height, az, el, sig)
        if ll > best_ll
            best_ll, best_az, best_el = ll, az, el
        end
    end
    best_az, best_el
end

const SIG_AZ = 0.001
const SIG_EL = 0.001

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
                       width::Int=256, height::Int=256, sig::Real=0.001)
    (tr, _) = mh(tr, ransac_proposal, (vertices, edges, obs_img, width, height, sig))
    for _ in 1:20
        tr = gaussian_drift_update(tr)
    end
    tr
end

function gaussian_drift_inference(vertices, edges, obs_img;
                                  width::Int=256, height::Int=256, sig::Real=0.001,
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
                          width::Int=256, height::Int=256, sig::Real=0.001,
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

visualize_trace(vertices, edges, tr; title="") = begin
    az = tr[:az]
    el = tr[:el]
    img, _ = render_wireframe_makie(vertices, edges; width=256, height=256, azimuth=az, elevation=el, perspectiveness=0.9)
    Plots.plot(img, axis=false, border=false, title=title)
end

function animate_drift(vertices, edges, tr; steps=500, path="vis.gif", fps=20)
    anim = Plots.@animate for i in 1:steps
        tr = gaussian_drift_update(tr)
        visualize_trace(vertices, edges, tr; title="Iteration $i/$steps")
    end
    Plots.gif(anim, path, fps=fps)
    tr
end

VD = Float32.([
    0.1411 0.0032 0.0000;
    0.4762 0.0049 0.0000;
    0.5785 0.2094 0.0000;
    0.4727 0.0016 0.0000;
    0.9965 0.2110 0.0000;
    0.5785 0.2110 0.0000;
    0.9894 0.7500 0.0000;
    0.9965 0.2127 0.0000;
    0.9365 0.7013 0.0000;
    0.9894 0.7532 0.0000;
    0.9453 0.2565 0.0000;
    1.0000 0.2110 0.0000;
    0.9383 0.6997 0.0000;
    0.9436 0.2516 0.0000;
    0.5802 0.2094 0.0000;
    0.5838 0.7581 0.0000;
    0.9912 0.7516 0.0000;
    0.5836 0.7572 0.0000;
    0.4497 0.7062 0.0000;
    0.4462 0.2516 0.0000;
    0.1429 0.0032 0.0000;
    0.4444 0.2532 0.0000;
    0.4709 0.0016 0.0000;
    0.4832 1.0000 0.0000;
    0.1693 0.9935 0.0000;
    0.1393 0.0000 0.0000;
    0.4515 0.7045 0.0000;
    0.4450 0.2512 0.0000;
    0.4533 0.7045 0.0000;
    0.4868 0.9968 0.0000;
    0.5838 0.7549 0.0000;
    0.1711 0.9951 0.0000;
    0.4850 0.9968 0.0000;
    0.0018 0.5276 0.0000;
    0.0000 0.5276 0.0000
])

ED = [(1,2),(3,4),(5,6),(7,8),(9,10),(9,19),(11,12),(13,14),(14,20),(15,16),(17,18),(21,22),(23,24),(25,26),(25,29),(27,28),(30,31),(32,33),(34,35)]

tr = ransac_inference(VD, ED, obs_img; width=256, height=256, sig=0.001, steps=500);
println("Estimated azimuth:", tr[:az]);
println("Estimated elevation:", tr[:el]);
# animate_drift(VD, ED, tr; steps=500, path="vis.gif", fps=20);

println("Best pose log-likelihood: ", global_best_ll[])

best_img, _ = render_wireframe_makie(VD, ED; width=256, height=256, azimuth=tr[:az], elevation=tr[:el], perspectiveness=0.9)
save("best_pose.png", best_img)
