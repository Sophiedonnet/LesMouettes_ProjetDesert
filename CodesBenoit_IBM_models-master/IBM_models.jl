Pkg.add(["StatsBase", "RCall", "Plots", "StatsPlots", "Random", "LinearAlgebra", "Distributions"])

using StatsBase, RCall, Plots, StatsPlots, Random, LinearAlgebra, Distributions


#region 1) Parameters 

function Get_parameters(model)

    if model == "Kefi" || model == "kefi"
        param = Param_Kefi()

    elseif model == "Guichard" || model == "guichard"
        param = Param_Guichard()

    elseif model == "Eby" || model == "eby" || model == "eby_feedback"
        param = Param_Eby()

    elseif model == "forest_gap" || model == "gap" || model == "tree"
        param = Param_Gap()

    else
        param = Param_Schneider()
    end
end




function Param_Guichard()

    δ = 0
    α2 = 0.3
    α0 = 0.4
    tau_leap = 0.5

    return Dict([
        "δ" => δ,
        "α2" => α2,
        "α0" => α0,
        "tau_leap" => tau_leap])
end



function Param_Gap()

    δ = 0.3
    d = 0.2
    α = 0.2
    tau_leap = 0.5

    return Dict([
        "δ" => δ,
        "d" => d,
        "α" => α,
        "tau_leap" => tau_leap])
end

function Param_Kefi()

    r = 0.0001
    d = 0.2
    f = 0.9
    m = 0.1
    b = 1
    c = 0.3
    delta = 0.1
    z = 4
    tau_leap = 0.5


    return Dict([
        "r" => r,
        "d" => d,
        "f" => f,
        "m" => m,
        "b" => b,
        "c" => c,
        "delta" => delta,
        "z" => z,
        "tau_leap" => tau_leap])
end


function Param_Schneider()

    r = 0.0001
    d = 0.2
    f = 0.9
    m = 0.05
    b = 0.3
    c = 0.3
    delta = 0.1
    g0 = 0.2
    z = 4
    tau_leap = 0.5


    return Dict([
        "r" => r,
        "d" => d,
        "f" => f,
        "m" => m,
        "b" => b,
        "c" => c,
        "delta" => delta,
        "z" => z,
        "g0" => g0,
        "tau_leap" => tau_leap])
end


function Param_Eby()

    p = 0.8
    q = 0.8

    return Dict([
        "p" => p,
        "q" => q])
end



#endregion







#landscapes


function Get_initial_lattice(model, ; size_landscape=100)

    if model == "Kefi" || model == "kefi" || model == "Schneider" || model == "schneider"
        ini_land = reshape(sample([-1, 0, 1], Weights([0.1, 0.1, 0.8]), size_landscape * size_landscape), size_landscape, size_landscape)
    elseif model == "Eby" || model == "eby" || model == "eby_feedback" || model == "forest_gap" || model == "gap" || model == "tree"
        ini_land = reshape(sample([1, 0], Weights([0.8, 0.2]), size_landscape * size_landscape), size_landscape, size_landscape)
    else
        ini_land = reshape(sample([0, 1, 2], Weights([0.4, 0.4, 0.2]), size_landscape * size_landscape), size_landscape, size_landscape)
    end

    return ini_land
end










#Runing model


function Run_model(; model, param, landscape, tmax=1000,
    keep_landscape=false, n_time_bw_snap=50, n_snapshot=25, intensity_feedback=1, burning_phase=1000) #in case we keep multiple landscapes at asymptotic state

    if model == "Kefi" || model == "kefi"

        dyn, land = IBM_Kefi_drylands(landscape=copy(landscape), param=copy(param), time_t=tmax,
            keep_landscape=keep_landscape, n_snapshot=n_snapshot, burning_phase=burning_phase, n_time_bw_snap=n_time_bw_snap)

    elseif model == "Guichard" || model == "guichard"

        dyn, land = IBM_Guichard_mussel(landscape=copy(landscape), param=copy(param), time_t=tmax,
            keep_landscape=keep_landscape, n_snapshot=n_snapshot, burning_phase=burning_phase, n_time_bw_snap=n_time_bw_snap)

    elseif model == "Eby" || model == "eby" || model == "eby_feedback"

        dyn, land = IBM_Eby_drylands(landscape=copy(landscape), param=copy(param), time_t=tmax,
            keep_landscape=keep_landscape, n_snapshot=n_snapshot, burning_phase=burning_phase,
            n_time_bw_snap=n_time_bw_snap, intensity_feedback=intensity_feedback)

    elseif model == "forest_gap" || model == "gap" || model == "tree"

        dyn, land = IBM_Forest_gap(landscape=copy(landscape), param=copy(param), time_t=tmax,
            keep_landscape=keep_landscape, n_snapshot=n_snapshot, burning_phase=burning_phase, n_time_bw_snap=n_time_bw_snap)

    else
        dyn, land = IBM_Schneider_drylands(landscape=copy(landscape), param=copy(param), time_t=tmax,
            keep_landscape=keep_landscape, n_snapshot=n_snapshot, burning_phase=burning_phase, n_time_bw_snap=n_time_bw_snap)
    end

    return dyn, land

end


















#region 2) Ploting functions





function Plot_dynamics(model, d)


    if model == "Guichard" || model == "guichard"

        plot(d[:, 2], label="Disturbed", color="#6D6D6D", linewidth=2)
        plot!(d[:, 3], label="Empty", color="#88BAC1", linewidth=2)
        plot!(d[:, 4], label="Mussel", color="#E89090", linewidth=2)
        ylims!((0.0, 1))

    elseif model == "Eby" || model == "eby" || model == "eby_feedback" || model == "forest_gap" || model == "gap" || model == "tree"

        plot(d[:, 1], seriescolor=:lightgreen, label="vegetation")
        ylims!((0.0, 1))

    else
        plot(d[:, 1], d[:, 2], seriescolor=:lightgreen, label="vegetation")
        plot!(d[:, 1], d[:, 3], seriescolor=:orange, label="fertile")
        plot!(d[:, 1], d[:, 4], seriescolor=:grey, label="degraded")
        ylims!((0.0, 1))
    end

end




function Plot_landscape(model, landscape, ; keep_fertile=true)


    if model == "Guichard" || model == "guichard"
        heatmap(landscape, color=["#6D6D6D", "#88BAC1", "#E89090"], axis=([], false), legend=false)
    elseif model == "Eby" || model == "eby" || model == "eby_feedback"
        colGRAD = cgrad([colorant"white", colorant"black"])
        heatmap(landscape, yflip=true, fill=true, c=colGRAD, legend=false)
    else
        if keep_fertile
            colGRAD = cgrad([colorant"white", colorant"gray", colorant"black"])
            heatmap(landscape, yflip=true, fill=true, c=colGRAD, legend=false)
        else
            landscape[findall((landscape .== 0))] .= -1
            colGRAD = cgrad([colorant"white", colorant"black"])
            heatmap(landscape, yflip=true, fill=true, c=colGRAD, legend=false)
        end
    end
end



#endregion



#region 3) Functions in details for each model



function select_neighbor(row, col, N)
    #Accounts for torus landscape (periodic boundaries)

    i, j = copy(row), copy(col)

    if i == 1
        top = N
    else
        top = i - 1
    end


    if i == N
        bottom = 1
    else
        bottom = i + 1
    end

    if j == N
        right = 1
    else
        right = j + 1
    end

    if j == 1
        left = N
    else
        left = j - 1
    end

    test = rand()

    col_neigh = j
    row_neigh = i


    if test <= 0.25
        row_neigh = top
    elseif test <= 0.5
        row_neigh = bottom
    elseif test <= 0.75
        col_neigh = left
    else
        col_neigh = right
    end


    return row_neigh, col_neigh

end



function select_neighbor_pair(coordinate_neighbors, Intensity_feedback)
    return convert(Array{Int64}, coordinate_neighbors[shuffle(1:end), :][1:Intensity_feedback, :])
end


function Get_coordinate(row, col, row_n, col_n, N)
    i, j = copy(row), copy(col)
    i_n, j_n = copy(row_n), copy(col_n)

    #Boundaries condition for focal site (i,j) 

    if i == 1
        top = N
    else
        top = i - 1
    end

    if i == N
        bottom = 1
    else
        bottom = i + 1
    end

    if j == N
        right = 1
    else
        right = j + 1
    end

    if j == 1
        left = N
    else
        left = j - 1
    end

    #Boundaries condition for its neighbor (i_n,j_n)
    if i_n == 1
        topn = N
    else
        topn = i_n - 1
    end

    if i_n == N
        bottomn = 1
    else
        bottomn = i_n + 1
    end

    if j_n == N
        rightn = 1
    else
        rightn = j_n + 1
    end

    if j_n == 1
        leftn = N
    else
        leftn = j_n - 1
    end



    coordinate_n = zeros(8, 2)
    coordinate_n[1, :] = [i right]
    coordinate_n[2, :] = [i left]
    coordinate_n[3, :] = [top j]
    coordinate_n[4, :] = [bottom j]
    coordinate_n[5, :] = [i_n rightn]
    coordinate_n[6, :] = [i_n leftn]
    coordinate_n[7, :] = [topn j_n]
    coordinate_n[8, :] = [bottomn j_n]

    #and filter the ones corresponding to focal site and its selected neighbor
    coordinate_n = coordinate_n[setdiff(1:8, (findall(coordinate_n[:, 1] .== i .&& coordinate_n[:, 2] .== j)[1],
        findall(coordinate_n[:, 1] .== i_n .&& coordinate_n[:, 2] .== j_n)[1])), :]

end






function IBM_Eby_drylands(; landscape, param, time_t, keep_landscape=false, n_snapshot=25, burning_phase=400, n_time_bw_snap=50, intensity_feedback=1)

    if keep_landscape
        all_landscape_snap = zeros(size(landscape)[1], size(landscape)[1], n_snapshot)
        nsave = 1
    end

    p_param = param["p"]
    q_param = param["q"]

    #If we keep all snapshots we determine the minimum time for having n_snapshot after a burning_phase and with n_time_bw_snap time step between each

    if keep_landscape
        time_t = burning_phase + n_time_bw_snap * n_snapshot
    end
    d2 = zeros(time_t, 2) #Allocating
    d2[1, :] = vec([sum(landscape) / size(landscape)[1]^2, 1 - sum(landscape) / size(landscape)[1]^2])

    @inbounds for t in 1:time_t

        #for each time step, we perform N**2 (N the dimension of the landscape) iterations to ensure that all sites get change on average 1 time per time step  
        @inbounds for focal_i in eachindex(1:size(landscape)[1]), focal_j in eachindex(1:size(landscape)[1])

            if landscape[focal_i, focal_j] == 1 #if vegetation otherwise do nothing

                neigh = select_neighbor(focal_i, focal_j, size(landscape)[1])

                if landscape[neigh[1], neigh[2]] == 0 #if neighbor is unoccupied
                    if rand() <= p_param #then there is reproduction in a neighbor
                        landscape[neigh[1], neigh[2]] = 1
                    else #else focal individual dies
                        landscape[focal_i, focal_j] = 0
                    end

                else #neighbor is occupied
                    if rand() <= q_param #facilitation from the neighbor
                        coord_neigh = Get_coordinate(focal_i, focal_j, neigh[1], neigh[2], size(landscape)[1])
                        neighbors = select_neighbor_pair(coord_neigh, intensity_feedback) #that is changed to an occupied cell
                        for i in eachindex(neighbors[:, 1])
                            landscape[neighbors[i, 1], neighbors[i, 2]] = 1
                        end

                    else
                        landscape[focal_i, focal_j] = 0
                    end
                end
            end #end loop for a focal cell

        end #loop on interactions

        d2[t, 1] = sum(landscape) / length(landscape)


        if keep_landscape && t > burning_phase && t % ((time_t - burning_phase) / n_snapshot) == 0 #for each time for which there is a snapshot, we save the landscape
            all_landscape_snap[:, :, nsave] = landscape
            nsave += 1
        end



    end

    d2[:, 2] = 1 .- d2[:, 1]

    if !keep_landscape
        all_landscape_snap = landscape
    end



    return d2, all_landscape_snap
end






function IBM_Kefi_drylands(; landscape, param, time_t, keep_landscape=false, n_snapshot=25, burning_phase=1000, n_time_bw_snap=50)

    if keep_landscape
        all_landscape_snap = zeros(size(landscape)[1], size(landscape)[1], n_snapshot)
        nsave = 1
    end
    r = param["r"]
    d = param["d"]
    f = param["f"]
    m = param["m"]
    b = param["b"]
    c = param["c"]
    delta = param["delta"]
    z = param["z"]
    tau_leap = param["tau_leap"]


    rules_change = transpose([0 1 -1 0; 1 0 0 -1])

    #Global densities
    rho_1 = length(findall((landscape .== 1))) / length(landscape) #fraction stress_tol

    nb_cell = size(landscape)[1]

    #Allocating 
    Rate_landscape = zeros(nb_cell, nb_cell, 4)

    #If we keep all snapshots we determine the minimum time for having n_snapshot after a burning_phase and with n_time_bw_snap time step between each

    if keep_landscape
        time_t = burning_phase + n_time_bw_snap * n_snapshot
    end
    d2 = zeros(time_t, 4) #Allocating



    for t in 1:time_t


        @rput landscape
        R"neigh_1= simecol::neighbors(x =landscape,state = 1, wdist =  matrix( c(0, 1, 0,1, 0, 1, 0, 1, 0), nrow = 3),bounds = 1)"
        @rget neigh_1

        Rate_landscape[:, :, 1] .= @. (delta * rho_1 + (1 - delta) * neigh_1 / z) * @.(b - c * rho_1)
        Rate_landscape[:, :, 1] .= Rate_landscape[:, :, 1] .* (landscape .== 0)

        Rate_landscape[:, :, 2] .= m .* (landscape .== 1)
        Rate_landscape[:, :, 3] .= @.(r + f * neigh_1 / z) .* (landscape .== -1)
        Rate_landscape[:, :, 4] .= d .* (landscape .== 0)

        Rate_landscape[findall(Rate_landscape .< 0)] .= 0 #to avoid problems with propensity

        #calculate propensity

        propensity = [sum(Rate_landscape[:, :, k]) for k in 1:size(Rate_landscape)[3]]

        nb_events = map(x -> rand(Poisson(x)), propensity * tau_leap)

        for event in 1:length(nb_events) #for each type of events
            patches = findall(landscape .== rules_change[event, 1])

            if nb_events[event] != 0 && length(patches) > nb_events[event]
                landscape[wsample(patches, Rate_landscape[patches, event], nb_events[event])] .= rules_change[event, 2]
            end
        end

        rho_1 = length(findall((landscape .== 1))) / length(landscape) #fraction vegetation
        rho_f = length(findall((landscape .== 0))) / length(landscape) #fraction fertile
        rho_d = 1 - rho_1 - rho_f # fraction degraded

        @views d2[t, :] = [t rho_1 rho_f rho_d]
        Rate_landscape = zeros(nb_cell, nb_cell, 4)

        #keeping the landscapes to average the summary statistics
        if keep_landscape && t > burning_phase && t % ((time_t - burning_phase) / n_snapshot) == 0
            all_landscape_snap[:, :, nsave] = landscape
            nsave += 1
        end

    end

    if !keep_landscape
        all_landscape_snap = landscape
    end

    return d2, all_landscape_snap
end



function IBM_Schneider_drylands(; landscape, param, time_t, keep_landscape=false, n_snapshot=25, burning_phase=1000, n_time_bw_snap=50)

    if keep_landscape
        all_landscape_snap = zeros(size(landscape)[1], size(landscape)[1], n_snapshot)
        nsave = 1
    end
    r = param["r"]
    d = param["d"]
    f = param["f"]
    m = param["m"]
    b = param["b"]
    c = param["c"]
    delta = param["delta"]
    g0 = param["g0"]
    z = param["z"]
    tau_leap = param["tau_leap"]


    rules_change = transpose([0 1 -1 0; 1 0 0 -1])

    #Global densities
    rho_1 = length(findall((landscape .== 1))) / length(landscape) #fraction stress_tol

    nb_cell = size(landscape)[1]

    #Allocating 
    Rate_landscape = zeros(nb_cell, nb_cell, 4)

    #If we keep all snapshots we determine the minimum time for having n_snapshot after a burning_phase and with n_time_bw_snap time step between each

    if keep_landscape
        time_t = burning_phase + n_time_bw_snap * n_snapshot
    end
    d2 = zeros(time_t, 4) #Allocating



    for t in 1:time_t


        @rput landscape
        R"neigh_1= simecol::neighbors(x =landscape,state = 1, wdist =  matrix( c(0, 1, 0,1, 0, 1, 0, 1, 0), nrow = 3),bounds = 1)"
        @rget neigh_1

        Rate_landscape[:, :, 1] .= @. (delta * rho_1 + (1 - delta) * neigh_1 / z) * @.(b - c * rho_1)
        Rate_landscape[:, :, 1] .= Rate_landscape[:, :, 1] .* (landscape .== 0)

        Rate_landscape[:, :, 2] .= @.(m + g0 * (1 - neigh_1 / z))
        Rate_landscape[:, :, 2] .= Rate_landscape[:, :, 2] .* (landscape .== 1)
        Rate_landscape[:, :, 3] .= @.(r + f * neigh_1 / z) .* (landscape .== -1)
        Rate_landscape[:, :, 4] .= d .* (landscape .== 0)

        Rate_landscape[findall(Rate_landscape .< 0)] .= 0 #to avoid problems with propensity

        #calculate propensity

        propensity = [sum(Rate_landscape[:, :, k]) for k in 1:size(Rate_landscape)[3]]

        nb_events = map(x -> rand(Poisson(x)), propensity * tau_leap)

        for event in 1:length(nb_events) #for each type of events
            patches = findall(landscape .== rules_change[event, 1])

            if nb_events[event] != 0 && length(patches) > nb_events[event]
                landscape[wsample(patches, Rate_landscape[patches, event], nb_events[event])] .= rules_change[event, 2]
            end
        end

        rho_1 = length(findall((landscape .== 1))) / length(landscape) #fraction vegetation
        rho_f = length(findall((landscape .== 0))) / length(landscape) #fraction fertile
        rho_d = 1 - rho_1 - rho_f # fraction degraded

        @views d2[t, :] = [t rho_1 rho_f rho_d]
        Rate_landscape = zeros(nb_cell, nb_cell, 4)

        #keeping the landscapes to average the summary statistics
        if keep_landscape && t > burning_phase && t % ((time_t - burning_phase) / n_snapshot) == 0
            all_landscape_snap[:, :, nsave] = landscape
            nsave += 1
        end

    end

    if !keep_landscape
        all_landscape_snap = landscape
    end

    return d2, all_landscape_snap
end


function IBM_Guichard_mussel(; landscape, param, time_t, keep_landscape=false, n_snapshot=25, burning_phase=1000, n_time_bw_snap=50)

    if keep_landscape
        all_landscape_snap = zeros(size(landscape)[1], size(landscape)[1], n_snapshot)
        nsave = 1
    end
    δ = param["δ"]
    α2 = param["α2"]
    α0 = param["α0"]
    tau_leap = param["tau_leap"]


    rules_change = transpose([0 1 2; 1 2 0])

    nb_cell = size(landscape)[1]

    #Allocating 
    Rate_landscape = zeros(nb_cell, nb_cell, 3)

    #If we keep all snapshots we determine the minimum time for having n_snapshot after a burning_phase and with n_time_bw_snap time step between each

    if keep_landscape
        time_t = burning_phase + n_time_bw_snap * n_snapshot
    end
    d2 = zeros(time_t, 4) #Allocating



    for t in 1:time_t


        @rput landscape
        R"neigh_2= simecol::neighbors(x =landscape,state = 2, wdist =  matrix( c(1, 1, 1,1, 0, 1, 1, 1, 1), nrow = 3),bounds = 1)"
        R"neigh_0= simecol::neighbors(x =landscape,state = 0, wdist =  matrix( c(1, 1, 1,1, 0, 1, 1, 1, 1), nrow = 3),bounds = 1)"
        @rget neigh_2
        @rget neigh_0



        Rate_landscape[:, :, 1] .= 1 .* (landscape .== 0)

        Rate_landscape[:, :, 2] .= @. (α2 * neigh_2 / 8)
        Rate_landscape[:, :, 2] .= Rate_landscape[:, :, 2] .* (landscape .== 1)

        Rate_landscape[:, :, 3] .= (α0 .* convert(Matrix{Float64}, neigh_0 .> 0) .+ δ)
        Rate_landscape[:, :, 3] .= Rate_landscape[:, :, 3] .* (landscape .== 2)

        Rate_landscape[findall(Rate_landscape .< 0)] .= 0 #to avoid problems with propensity

        #calculate propensity

        propensity = [sum(Rate_landscape[:, :, k]) for k in 1:size(Rate_landscape)[3]]

        nb_events = map(x -> rand(Poisson(x)), propensity * tau_leap)

        for event in 1:length(nb_events) #for each type of events
            patches = findall(landscape .== rules_change[event, 1])

            if nb_events[event] != 0 && length(patches) > nb_events[event]
                landscape[wsample(patches, Rate_landscape[patches, event], nb_events[event])] .= rules_change[event, 2]
            end
        end

        rho_0 = length(findall((landscape .== 0))) / length(landscape) #fraction disturbed
        rho_1 = length(findall((landscape .== 1))) / length(landscape) #fraction empty
        rho_2 = 1 - rho_0 - rho_1 # fraction occupied

        @views d2[t, :] = [t rho_0 rho_1 rho_2]
        Rate_landscape = zeros(nb_cell, nb_cell, 3)

        #keeping the landscapes to average the summary statistics
        if keep_landscape && t > burning_phase && t % ((time_t - burning_phase) / n_snapshot) == 0
            all_landscape_snap[:, :, nsave] = landscape
            nsave += 1
        end

    end

    if !keep_landscape
        all_landscape_snap = landscape
    end

    return d2, all_landscape_snap


end



function IBM_Forest_gap(; landscape, param, time_t, keep_landscape=false, n_snapshot=25, burning_phase=1000, n_time_bw_snap=50)

    if keep_landscape
        all_landscape_snap = zeros(size(landscape)[1], size(landscape)[1], n_snapshot)
        nsave = 1
    end
    δ = param["δ"]
    d = param["d"]
    α = param["α"]
    tau_leap = param["tau_leap"]


    rules_change = transpose([0 1; 1 0])

    nb_cell = size(landscape)[1]

    #Allocating 
    Rate_landscape = zeros(nb_cell, nb_cell, 2)

    #If we keep all snapshots we determine the minimum time for having n_snapshot after a burning_phase and with n_time_bw_snap time step between each

    if keep_landscape
        time_t = burning_phase + n_time_bw_snap * n_snapshot
    end
    d2 = zeros(time_t, 3) #Allocating



    for t in 1:time_t


        @rput landscape
        R"neigh_0= simecol::neighbors(x =landscape,state = 0, wdist = matrix( c(0, 1, 0,1, 0, 1, 0, 1, 0), nrow = 3),bounds = 1)"
        @rget neigh_0



        Rate_landscape[:, :, 1] .= α .* (landscape .== 0)

        Rate_landscape[:, :, 2] .= @. (d + δ * neigh_0 / 4)
        Rate_landscape[:, :, 2] .= Rate_landscape[:, :, 2] .* (landscape .== 1)

        Rate_landscape[findall(Rate_landscape .< 0)] .= 0 #to avoid problems with propensity

        #calculate propensity

        propensity = [sum(Rate_landscape[:, :, k]) for k in 1:size(Rate_landscape)[3]]

        nb_events = map(x -> rand(Poisson(x)), propensity * tau_leap)

        for event in 1:length(nb_events) #for each type of events
            patches = findall(landscape .== rules_change[event, 1])

            if nb_events[event] != 0 && length(patches) > nb_events[event]
                landscape[wsample(patches, Rate_landscape[patches, event], nb_events[event])] .= rules_change[event, 2]
            end
        end

        rho_0 = length(findall((landscape .== 0))) / length(landscape) #fraction empty
        rho_1 = length(findall((landscape .== 1))) / length(landscape) #fraction forest

        @views d2[t, :] = [t rho_0 rho_1]
        Rate_landscape = zeros(nb_cell, nb_cell, 2)

        #keeping the landscapes to average the summary statistics
        if keep_landscape && t > burning_phase && t % ((time_t - burning_phase) / n_snapshot) == 0
            all_landscape_snap[:, :, nsave] = landscape
            nsave += 1
        end

    end

    if !keep_landscape
        all_landscape_snap = landscape
    end

    return d2, all_landscape_snap


end

#endregion
