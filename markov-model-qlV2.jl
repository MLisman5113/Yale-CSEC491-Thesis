# ####################################################################
# markov-model-qlV2.jl
#
# A file implementing Q-learning for the two-state 
# Markov productivity environment. We assume that "ModelParams" and 
# basic environment definitions are accessible (from markov-model-final.jl).
#
# Also generates plots showing the results.
#
# Boltzmann (softmax) exploration is used for action selection.
# We demonstrate how to:
#   1) Run Q-learning for "no hostility" (s_bar >= k => 0 hostility),
#   2) Run Q-learning for "with hostility" (s_bar < k => hostility cost > 0),
# 
#   To run this file in Julia REPL:
#   1) include("markov-model-final.jl")
#   2) include("markov-model-qlV2.jl")
#   3) MarkovModelQL.demo_run()
#
#   Author: Marcus A. Lisman, Yale University
#   Written for CSEC 491 Senior Thesis
#
# ####################################################################

module MarkovModelQL

using Random, Printf, Statistics, Plots, Distributions
using CSV, DataFrames
using ..MarkovModelFinal: ModelParams, hostility_cost, print_solution
using Interpolations
using PrettyTables

# ####################################################################
# 0.  House‑keeping constants & helpers                              #
# ####################################################################

# Figures I want to keep
const KEEP_FIGS = Set([
    "firing_comparison.png",
    "flip_threshold.png",
    "hostility_incidence.png",
    "payoff_comparison.png",
    "payoff_gap.png",
    "profit_surplus.png",
    "reward_variance.png",
    "surplus_ratio.png",
    "total_welfare.png",
    "worker_payoff.png",
    "profit_heat_unknown_kq.png",
    "fire_heat_unknown_kq.png",
    "profit_unknown_p.png",
    "fire_unknown_p.png",
])
const N_AVG = 1000

# Ensure output folders exist
mkpath("plots")
mkpath("data")

############# CSV helper ##################
#=
    save_to_csv(fname; kw...)

Append vectors stored in keyword arguments to `fname` as a DataFrame.
If the file is absent it is created with headers.
=#
function save_to_csv(fname; kw...)
    df = DataFrame(; kw...)
    CSV.write(fname, df; append=isfile(fname))
end

############## Plot gate helper ##################
#=
    maybe_plot(fname::AbstractString, plot_func, args...; kwargs...)

Run `plot_func(args...; filename="plots/$fname", kwargs...)`
*only* if `fname ∈ KEEP_FIGS`.
=#

function maybe_plot(fname, plot_func, args...; kwargs...)
    if fname in KEEP_FIGS
        kwargs = merge(Dict(:filename=>"plots/$fname"), kwargs)
        plot_func(args...; kwargs...)
    end
    return nothing
end

function avg_qmetric(f::Function, params; N=15)
    tmp = [f(run_q_learning(params; rng=MersenneTwister(i),
                            num_episodes=num_ep, max_steps=ms)...)
           for i in 1:N]
    return mean(tmp)
end

# ####################################################################
# 1. Environment Setup for Q-Learning
# ####################################################################

"""
We define the firm 'environment' states as:

  (empStatus, alpha) ∈ { (E, H), (E, L), (U, H), (U, L) }

Possible actions:
  - If (E, H): keep or fire
  - If (E, L): keep or fire
  - If (U, H): hire or wait
  - If (U, L): wait  (only 1 feasible action: do nothing / remain unemployed)

We'll index states and actions for Q-table as integers for convenience:

We can define:
   state_index(E,H) = 1
   state_index(E,L) = 2
   state_index(U,H) = 3
   state_index(U,L) = 4

   action_index(keep) = 1
   action_index(fire) = 2
   action_index(hire) = 1
   action_index(wait) = 2

We'll store Q[state, action].
"""

# define enumerations
@enum EmpStatus begin
    E  # employed
    U  # unemployed
end

@enum ProdState begin
    High  # H
    Low   # L
end

const ALL_STATES = [(E, High), (E, Low), (U, High), (U, Low)]


# Ensure the output folder exists:
mkpath("plots")

state_index(emp::EmpStatus, α::ProdState)::Int = 
    emp==E && α==High ? 1 :
    emp==E && α==Low  ? 2 :
    emp==U && α==High ? 3 : 4

action_space(emp::EmpStatus, α::ProdState)::Vector{Symbol} = emp==E ? [:keep, :fire] :
    (α==High ? [:hire, :wait] : [:wait])

action_index(a::Symbol)::Int = a===:keep  ? 1 :
                             a===:fire  ? 2 :
                             a===:hire  ? 1 :
                             a===:wait  ? 2 : error("Unknown action $a")

# ####################################################################
# 2. Softmax & Sampling
# ####################################################################

"""
    softmax(vals, τ)

Boltzmann probabilities over `vals` at temperature τ.
"""
function softmax(vals::AbstractVector{T}, τ::T) where T<:Real
    if τ ≤ 1e-9
        probs = zero.(vals)
        probs[argmax(vals)] = one(T)
        return probs
    else
        exps = exp.(vals ./ τ)
        return exps ./ sum(exps)
    end
end

"""
    sample_from_probs(probs, rng)

Sample an index i with probability = probs[i].
"""
function sample_from_probs(probs::AbstractVector{T}, rng::AbstractRNG=Random.GLOBAL_RNG) where T<:Real
    u = rand(rng)
    cum = cumsum(probs)
    return searchsortedfirst(cum, u)
end


# ####################################################################
# 3. Environment Step
# ####################################################################

function env_step(params::ModelParams, emp::EmpStatus, α::ProdState, a::Symbol)
    p, w, s̄, k, δ, H, L = params.p, params.w, params.s_bar, params.k, params.δ, params.H, params.L
    h = max(0, k - s̄)

    # reward for the firm
    r = emp==E ? (a==:keep ? (α==High ? H-w : L-w) : -h) :
        (emp==U && α==High && a==:hire ? (H-w) : 0.0)

    # next employment status
    next_emp = emp==E ? (a==:fire ? U : E) : (α==High && a==:hire ? E : U)

    # alpha transition
    next_α = rand() < p ? α : (α==High ? Low : High)

    return (r, next_emp, next_α)
end

# ####################################################################
# 4. Q‑Learning
# ####################################################################

"""
    run_q_learning(params; ...)

Returns:
  Q, rewards, surplus, q_diff, fired_flag, fire_frac, temps, entropy
"""
function run_q_learning(
    params::ModelParams;
    num_episodes::Int=100,
    max_steps::Int=2,
    alpha_Q::Float64=0.1,
    gamma::Float64=0.95,
    temperature::Float64=1.0,
    temp_decay::Float64=0.999,
    rng::AbstractRNG=Random.GLOBAL_RNG
)
    Q = zeros(4,2)
    rewards  = zeros(num_episodes)
    surplus  = zeros(num_episodes)
    q_diff   = zeros(num_episodes)
    fired    = falses(num_episodes)
    fire_frac= zeros(num_episodes)
    temps    = zeros(num_episodes)
    entropy  = zeros(num_episodes)

    for ep in 1:num_episodes
        Q_old = copy(Q)
        total_r = 0.0
        total_w = 0.0
        did_fire = false
        fire_count = 0

        τ = temperature * (temp_decay^(ep-1))
        temps[ep] = τ

        # entropy at (E,High)
        acts0 = action_space(E, High)
        vals0 = [Q[state_index(E,High), action_index(a)] for a in acts0]
        ps0   = softmax(vals0, τ)
        entropy[ep] = -sum(p>0 ? p*log(p) : 0.0 for p in ps0)

        emp, α = E, High

        for step in 1:max_steps
            sidx = state_index(emp, α)
            acts = action_space(emp, α)
            vals = [Q[sidx, action_index(a)] for a in acts]
            ps   = softmax(vals, τ)

            ai = sample_from_probs(ps, rng)
            a  = acts[ai]

            (r, emp2, α2) = env_step(params, emp, α, a)

            # track worker payout
            if a in (:keep, :hire)
                total_w += params.w
            end

            # track firing events
            if a==:fire
                did_fire = true
                fire_count += 1
            end

            total_r += r

            # Q‑update
            next_idx = state_index(emp2, α2)
            next_actions = action_space(emp2, α2)
            maxQ_next = maximum([Q[next_idx, action_index(a2)] for a2 in next_actions])
            td = r + gamma*maxQ_next - Q[sidx, action_index(a)]
            Q[sidx, action_index(a)] += alpha_Q * td

            emp, α = emp2, α2
        end

        rewards[ep]   = total_r
        surplus[ep]   = total_r + total_w
        q_diff[ep]    = maximum(abs.(Q .- Q_old))
        fired[ep]     = did_fire
        fire_frac[ep] = fire_count / max_steps
    end

    return (Q, rewards, surplus, q_diff, fired, fire_frac, temps, entropy)
end

# ####################################################################
# 5. Policy Extraction
# ####################################################################

function derive_policy(Q)
    pol = Dict{Tuple{EmpStatus,ProdState},Symbol}()
    for (i,st) in enumerate(ALL_STATES)
        acts = action_space(st...)
        qv   = [Q[i, action_index(a)] for a in acts]
        pol[st] = acts[argmax(qv)]
    end
    return pol
end

#####################################################################
#    Print Summary for Q‑learning                               
#####################################################################
"""
    print_qlearning_summary(params, Q, rewards, surplus, fired)

Console report analogous to DP’s `print_solution`.
"""
function print_qlearning_summary(
    params::ModelParams,
    Q::AbstractMatrix,
    rewards::AbstractVector,
    surplus::AbstractVector,
    fired::AbstractVector
)
    pol = derive_policy(Q)
    println("--------------------------------------------------------------")
    println(" Q‑LEARNING SUMMARY")
    @printf("   Episodes              : %d\n", length(rewards))
    @printf("   Avg firm reward       : %.4f\n", mean(rewards))
    @printf("   Avg social surplus    : %.4f\n", mean(surplus))
    @printf("   Firing incidence      : %.2f %%\n", 100*mean(fired))
    println("   Learned greedy policy:")
    for st in ALL_STATES
        println("     ", st, " → ", pol[st])
    end
    println("--------------------------------------------------------------")
end

"""
    compare_dp_rl(params; num_runs=10, num_ep=200, max_steps=2)

Prints the DP solution and an averaged Q‑learning summary
for the same parameters.
"""
function compare_dp_rl(params::ModelParams;
                       num_runs::Int=10,
                       num_ep::Int=200,
                       max_steps::Int=2)

    # ---------------- DP -----------------------------------------------------
    println("\n====  DP (exact)  ===============================================")
    dp_sol = value_iteration(params)
    print_solution(params, dp_sol)

    # ---------------- Q‑learning (averaged over seeds) -----------------------
    println("\n====  Q‑LEARNING  (average of $num_runs runs)  ===================")
    rewards = Float64[]; surplus = Float64[]; fired = Float64[]
    Q_accum = zeros(4,2)

    for seed in 1:num_runs
        rng = MersenneTwister(seed)
        (Q, r, s, _, f, _, _, _) = run_q_learning(
            params;
            num_episodes = num_ep,
            max_steps    = max_steps,
            rng          = rng               # deterministic seed
        )
        Q_accum .+= Q
        append!(rewards, r)
        append!(surplus, s)
        append!(fired,   f)
    end

    Q_mean = Q_accum ./ num_runs
    print_qlearning_summary(params, Q_mean, rewards, surplus, fired)
end


# # ####################################################################
# 3.  Plotting                                                         #
# # ####################################################################

function plot_profit_surplus(pvals, profits, surpluses; filename)
    plt = plot(pvals, profits, label="Firm profit", lw=2,
               xlabel="p", ylabel="Value", title="Profit vs Social Surplus")
    plot!(plt, pvals, surpluses, label="Social surplus", lw=2)
    savefig(plt, filename); println("saved → $filename"); plt
end

function plot_hostility_incidence(incid, pvals; filename)
    plt = plot(pvals, incid, lw=2,
               xlabel="p", ylabel="Frac episodes with fire",
               title="Hostility Incidence")
    savefig(plt, filename); println("saved → $filename"); plt
end

function plot_payoff_comparison(
    pvals, firm_noh, worker_noh, firm_h, worker_h; filename)
    plt = plot(pvals, firm_noh, label="Firm (no hostility)",
               xlabel="p", ylabel="Payoff",
               title="Firm & Worker Payoff", lw=2)
    plot!(plt, pvals, firm_h,   label="Firm (hostility)", lw=2)
    plot!(plt, pvals, worker_noh, linestyle=:dash,
          label="Worker (no hostility)", lw=2)
    plot!(plt, pvals, worker_h,  linestyle=:dash,
          label="Worker (hostility)", lw=2)
    savefig(plt, filename); println("saved → $filename"); plt
end

function plot_firing_comparison(pvals, fire_noh, fire_h; filename)
    plt = plot(pvals, fire_noh, label="Firing % (no hostility)",
               xlabel="p", ylabel="Fraction",
               title="Firing Fraction", lw=2)
    plot!(plt, pvals, fire_h, label="Firing % (hostility)", lw=2)
    savefig(plt, filename); println("saved → $filename"); plt
end

function plot_surplus_ratio(pvals, ratio_noh, ratio_h; filename)
    plt = plot(pvals, ratio_noh, label="No hostility", lw=2,
               xlabel="p", ylabel="Worker / Firm",
               title="Surplus Ratio")
    plot!(plt, pvals, ratio_h, label="Hostility", lw=2)
    savefig(plt, filename); println("saved → $filename"); plt
end

function plot_reward_variance(pvals, var_noh, var_h; filename)
    plt = plot(pvals, var_noh, label="No hostility", lw=2,
               xlabel="p", ylabel="Var(reward)",
               title="Reward Variance")
    plot!(plt, pvals, var_h, label="Hostility", lw=2)
    savefig(plt, filename); println("saved → $filename"); plt
end

function plot_worker_payoff(pvals, wpay_noh, wpay_h; filename)
    plt = plot(pvals, wpay_noh, label="Worker (no hostility)",
               xlabel="p", ylabel="Payoff",
               title="Worker Payoff", lw=2)
    plot!(plt, pvals, wpay_h, label="Worker (hostility)", lw=2)
    savefig(plt, filename); println("saved → $filename"); plt
end

function plot_payoff_gap(pvals, gap_noh, gap_h; filename)
    plt = plot(pvals, gap_noh, label="Gap (no hostility)",
               xlabel="p", ylabel="|Firm – Worker|",
               title="Payoff Gap", lw=2)
    plot!(plt, pvals, gap_h, label="Gap (hostility)", lw=2)
    savefig(plt, filename); println("saved → $filename"); plt
end

function plot_total_welfare(pvals, tot_noh, tot_h; filename)
    plt = plot(pvals, tot_noh, label="Welfare (no hostility)",
               xlabel="p", ylabel="Total payoff",
               title="Social Welfare", lw=2)
    plot!(plt, pvals, tot_h, label="Welfare (hostility)", lw=2)
    savefig(plt, filename); println("saved → $filename"); plt
end

function plot_flip_threshold(p_noh, p_h; filename)
    plt = scatter([p_noh, p_h], [0, 0],
                  label=["No hostility" "Hostility"],
                  xlabel="p", yticks=false,
                  title="Critical p for policy flip", markersize=8)
    savefig(plt, filename); println("saved → $filename"); plt
end


# ####################################################################
#    Value Iteration for V_E(α) and V_U(α)
# ####################################################################

"""
    value_iteration(params::ModelParams; tol=1e-8, maxiter=1000) 
        -> (VE_H, VE_L, VU_H, VU_L, policyE_H, policyE_L, policyU_H)

Solves the firm's infinite-horizon dynamic program using value iteration. 

Returns a tuple:
- VE_H : V_E(H)
- VE_L : V_E(L)
- VU_H : V_U(H)
- VU_L : V_U(L)
- policyE_H : either :keep or :fire in state (E,H)
- policyE_L : either :keep or :fire in state (E,L)
- policyU_H : either :hire or :wait in state (U,H)

(The policy in state (U,L) is always "wait" = do not hire, because it would be 
unprofitable to hire at L < w.)
"""
function value_iteration(params::ModelParams; tol=1e-8, maxiter=1000)

    # Extract for convenience
    δ = params.δ
    p = params.p
    w = params.w
    H_ = params.H
    L_ = params.L
    hcost = hostility_cost(params)  # k - s_bar if s_bar < k, else 0

    # Initialize guesses for value function
    # Track: V_E(H), V_E(L), V_U(H), V_U(L)
    VE_H, VE_L, VU_H, VU_L = 0.0, 0.0, 0.0, 0.0

    # We'll iterate until convergence
    for iter in 1:maxiter
        # Backup copies of old values
        old_VE_H, old_VE_L = VE_H, VE_L
        old_VU_H, old_VU_L = VU_H, VU_L

        # 1. Update V_U(L):
        #    If unemployed & α=L, can't profitably hire => payoff=0 
        #    plus δ * expectation of next state's V_U
        #    Next state α' = L with prob p, or H with prob (1-p)
        new_VU_L = 0.0 + δ * (p*old_VU_L + (1-p)*old_VU_H)

        # 2. Update V_U(H):
        #    If unemployed & α=H, two choices:
        #       (a) remain unemployed => immediate payoff=0 
        #           plus δ * E[next state's V_U]
        #       (b) hire => payoff=(H-w) + δ * E[next state's V_E]
        #    We take the max of these
        remain_unemployed_value = 0.0 + δ * (p*old_VU_H + (1-p)*old_VU_L)
        hire_value              = (H_ - w) + δ * (p*old_VE_H + (1-p)*old_VE_L)
        new_VU_H = max(remain_unemployed_value, hire_value)

        # 3. Update V_E(H):
        #    If employed & α=H, two choices:
        #       (a) keep => payoff=(H-w) + δ * E[next state's V_E]
        #       (b) fire => payoff = -hcost + δ * E[next state's V_U]
        keep_value_H = (H_ - w) + δ * (p*old_VE_H + (1-p)*old_VE_L)
        fire_value_H = -hcost + δ * (p*old_VU_H + (1-p)*old_VU_L)
        new_VE_H = max(keep_value_H, fire_value_H)

        # 4. Update V_E(L):
        #    If employed & α=L, two choices:
        #       (a) keep => payoff=(L-w) + δ * E[next state's V_E]
        #       (b) fire => payoff=-hcost + δ * E[next state's V_U]
        keep_value_L = (L_ - w) + δ * (p*old_VE_L + (1-p)*old_VE_H)
        fire_value_L = -hcost + δ * (p*old_VU_L + (1-p)*old_VU_H)
        new_VE_L = max(keep_value_L, fire_value_L)

        # Check convergence
        diff = maximum(abs.([
            new_VE_H - VE_H, 
            new_VE_L - VE_L, 
            new_VU_H - VU_H, 
            new_VU_L - VU_L
        ]))

        # Update
        VE_H, VE_L, VU_H, VU_L = new_VE_H, new_VE_L, new_VU_H, new_VU_L

        if diff < tol
            # We consider convergence
            # println("Value iteration converged in $(iter) iterations.")
            break
        end
    end

    # After convergence, extract the policy by seeing which choice is better
    # for each employed/unemployed state.

    # For (U,L) the policy is always "wait" because new_VU_L = 0 + δE[VU], 
    # and hiring is (L-w) + δE[VE], which is typically negative if L < w.
    policyU_L = :wait

    # For (U,H):
    # Compare remain_unemployed_value vs. hire_value
    remain_unemployed_value = 0.0 + δ * (p*VU_H + (1-p)*VU_L)
    hire_value              = (H_ - w) + δ * (p*VE_H + (1-p)*VE_L)
    policyU_H = remain_unemployed_value >= hire_value ? :wait : :hire

    # For (E,H):
    keep_value_H = (H_ - w) + δ * (p*VE_H + (1-p)*VE_L)
    fire_value_H = -hcost + δ * (p*VU_H + (1-p)*VU_L)
    policyE_H = keep_value_H >= fire_value_H ? :keep : :fire

    # For (E,L):
    keep_value_L = (L_ - w) + δ * (p*VE_L + (1-p)*VE_H)
    fire_value_L = -hcost + δ * (p*VU_L + (1-p)*VU_H)
    policyE_L = keep_value_L >= fire_value_L ? :keep : :fire

    return (VE_H, VE_L, VU_H, VU_L,
            policyE_H, policyE_L, policyU_H)
end

# # ####################################################################
#    DP-vs-Q-learning policy comparison for p = 0.50 : 0.05 : 0.99   
# # ####################################################################
function compare_dp_q_policies(params::ModelParams;
                               ps=collect(0.5:0.05:0.99),
                               num_ep::Int=2000,
                               max_steps::Int=20,
                               num_runs::Int=30)

    comparison_tables = Dict{Float64, DataFrame}()
    state_labels = Dict(
        (E, High) => "Employed, α = H",
        (E, Low)  => "Employed, α = L",
        (U, High) => "Unemployed, α = H",
        (U, Low)  => "Unemployed, α = L"
    )

    for p in ps
        # Update model parameters
        par = ModelParams(params.w, params.s_bar, params.k, p, params.δ, params.H, params.L)

        # ---------------- DP policy ----------------
        _, _, _, _, pol_E_H, pol_E_L, pol_U_H = value_iteration(par)
        dp_policy = Dict(
            (E, High) => pol_E_H,
            (E, Low)  => pol_E_L,
            (U, High) => pol_U_H,
            (U, Low)  => :wait
        )

        # ---------------- Q-learning policy (averaged) ----------------
        Q_accum = zeros(4,2)
        for seed in 1:num_runs
            rng = MersenneTwister(seed)
            Q, _, _, _, _, _, _, _ = run_q_learning(
                par; rng=rng,
                num_episodes=num_ep, max_steps=max_steps
            )
            Q_accum .+= Q
        end
        Q_avg = Q_accum ./ num_runs
        q_policy = derive_policy(Q_avg)

        # ---------------- Assemble comparison table ----------------
        rows = Vector{NamedTuple}()
        for st in ALL_STATES
            push!(rows, (
                State     = state_labels[st],
                DP_Action = dp_policy[st],
                Q_Action  = q_policy[st]
            ))
        end
        df = DataFrame(rows)
        comparison_tables[p] = df

        println("\n■ Policy comparison for p = $(round(p, digits=2))")
        pretty_table(df; tf=tf_unicode, alignment=:l)
    end

    return comparison_tables
end

function demo_run()
    # baseline parameters
    w, s̄, k, δ, H, L = 1.0, 0.5, 2.0, 0.95, 2.5, 0.5  # baseline 0.5 for severance
    num_ep, ms = 100, 2
    αQ, γ, τ0, decay = 0.1, 0.95, 1.0, 0.999

        """
            avg_metric(f, params; episodes=num_ep, steps=ms)
    
        Returns the average of `f(run_q_learning(...))` across `N_AVG` seeds.
        `f` is a function that extracts the statistic you want from the
        8‑tuple returned by `run_q_learning`.
        """
        function avg_metric(f::Function, params::ModelParams;
                            episodes::Int=num_ep, steps::Int=ms)
            vals = Float64[]
            for seed in 1:N_AVG
                rng = MersenneTwister(seed)
                stats = run_q_learning(params;
                         rng=rng, num_episodes=episodes, max_steps=steps,
                         alpha_Q=αQ, gamma=γ, temperature=τ0, temp_decay=decay)
                push!(vals, f(stats))
            end
            return mean(vals)
        end
    
    pvals = collect(0.5:0.05:0.99)

    # --------------------------------------------------------------#
    # 1)  Profit & Surplus sweep                                    #
    # --------------------------------------------------------------#
    profits    = [avg_metric(x -> mean(x[2]),    # 2 = rewards
                            ModelParams(w, s̄, k, p, δ, H, L))
                for p in pvals]

    surpluses  = [avg_metric(x -> mean(x[3]),    # 3 = surplus
                            ModelParams(w, s̄, k, p, δ, H, L))
                for p in pvals]

    maybe_plot("profit_surplus.png", plot_profit_surplus, pvals,
               profits, surpluses)
    save_to_csv("data/profit_surplus.csv";
        p = pvals, firm_profit = profits, social_surplus = surpluses)

    # --------------------------------------------------------------#
    # 2)  Hostility incidence                                       #
    # --------------------------------------------------------------#
    incid = [avg_metric(x -> mean(x[5]),          # 5 = fired (Bool vector)
                        ModelParams(w, s̄-1e-6, k, p, δ, H, L))
             for p in pvals]

    maybe_plot("hostility_incidence.png", plot_hostility_incidence,
               incid, pvals)
    save_to_csv("data/hostility_incidence.csv"; p = pvals, fired_frac = incid)

    # --------------------------------------------------------------#
    # 3)  Hostility vs No‑hostility comparisons                     #
    # --------------------------------------------------------------#
    n = length(pvals)
    firm_noh = zeros(n); worker_noh = zeros(n); fire_noh = zeros(n)
    firm_h   = zeros(n); worker_h   = zeros(n); fire_h   = zeros(n)

    for (i,p) in enumerate(pvals)
        params_noh = ModelParams(w, max(k,s̄), k, p, δ, H, L)
        firm_noh[i] = avg_metric(
            x -> mean(x[2]),                     # rewards
            params_noh)

         # Worker payoff = surplus – reward
        worker_noh[i] = avg_metric(
            x -> mean(x[3]) - mean(x[2]),        # 3 – 2
            params_noh)

        # Firing fraction
        fire_noh[i] = avg_metric(
            x -> mean(x[5]),                     # fired Bool
            params_noh)

        params_h = ModelParams(w, min(k,s̄-1e-6), k, p, δ, H, L)
        
        firm_h[i] = avg_metric(
            x -> mean(x[2]),                     # rewards
            params_h)

         # Worker payoff = surplus – reward
        worker_h[i] = avg_metric(
            x -> mean(x[3]) - mean(x[2]),        # 3 – 2
            params_h)

        # Firing fraction
        fire_h[i] = avg_metric(
            x -> mean(x[5]),                     # fired Bool
            params_h)
    end

    maybe_plot("payoff_comparison.png", plot_payoff_comparison,
               pvals, firm_noh, worker_noh, firm_h, worker_h)
    maybe_plot("firing_comparison.png", plot_firing_comparison,
               pvals, fire_noh, fire_h)

    save_to_csv("data/payoff_comparison.csv";
        p = pvals,
        firm_noh = firm_noh, worker_noh = worker_noh,
        firm_h   = firm_h,   worker_h   = worker_h)

    save_to_csv("data/firing_comparison.csv";
        p = pvals, fire_noh = fire_noh, fire_h = fire_h)

    # -----------------------------------------------------------#
    # 4)  Additional line plots                                  #
    # -----------------------------------------------------------#
    ratio_noh = worker_noh ./ firm_noh
    ratio_h   = worker_h   ./ firm_h
    var_noh   = fire_noh 
    var_h     = fire_h

    # variance in episode reward (need separate sweep)
    var_noh .= 0;  var_h .= 0
    for (i,p) in enumerate(pvals)
        params_noh = ModelParams(w, max(k,s̄), k, p, δ, H, L)
        (_, r_noh, _, _, _, _, _, _) = run_q_learning(params_noh;
            num_episodes=num_ep, max_steps=ms,
            alpha_Q=αQ, gamma=γ, temperature=τ0, temp_decay=decay)
        params_h = ModelParams(w, min(k,s̄-1e-6), k, p, δ, H, L)
        (_, r_h, _, _, _, _, _, _) = run_q_learning(params_h;
            num_episodes=num_ep, max_steps=ms,
            alpha_Q=αQ, gamma=γ, temperature=τ0, temp_decay=decay)
        var_noh[i] = var(r_noh)
        var_h[i]   = var(r_h)
    end

    maybe_plot("surplus_ratio.png",   plot_surplus_ratio,
               pvals, ratio_noh, ratio_h)
    maybe_plot("reward_variance.png", plot_reward_variance,
               pvals, var_noh, var_h)

    save_to_csv("data/surplus_ratio.csv";
        p = pvals, ratio_noh = ratio_noh, ratio_h = ratio_h)
    save_to_csv("data/reward_variance.csv";
        p = pvals, var_noh = var_noh, var_h = var_h)

    # -----------------------------------------------------------#
    # 5)  Worker payoff & welfare & gap                          #
    # -----------------------------------------------------------#
    gap_noh = abs.(firm_noh .- worker_noh)
    gap_h   = abs.(firm_h   .- worker_h)
    tot_noh = firm_noh .+ worker_noh
    tot_h   = firm_h   .+ worker_h

    maybe_plot("worker_payoff.png",   plot_worker_payoff,
               pvals, worker_noh, worker_h)
    maybe_plot("payoff_gap.png",      plot_payoff_gap,
               pvals, gap_noh, gap_h)
    maybe_plot("total_welfare.png",   plot_total_welfare,
               pvals, tot_noh, tot_h)

    save_to_csv("data/worker_payoff.csv";
        p = pvals, worker_noh = worker_noh, worker_h = worker_h)
    save_to_csv("data/payoff_gap.csv";
        p = pvals, gap_noh = gap_noh, gap_h = gap_h)
    save_to_csv("data/total_welfare.csv";
        p = pvals, welfare_noh = tot_noh, welfare_h = tot_h)

    # ---------------------------------------------------------------#
    # 6)  Flip‑threshold (critical p where policy switches in (E,L)) #
    # ---------------------------------------------------------------#
    function first_fire(params_template)
        for p in pvals
            par = params_template(p)
            Q, = run_q_learning(par; num_episodes=100, max_steps=ms)
            pol = derive_policy(Q)
            pol[(E,Low)] == :fire && return p
        end
        return NaN
    end
    p_noh = first_fire(p -> ModelParams(w, max(k,s̄), k, p, δ, H, L))
    p_h   = first_fire(p -> ModelParams(w, min(k,s̄-1e-6), k, p, δ, H, L))

    maybe_plot("flip_threshold.png", plot_flip_threshold, p_noh, p_h)
    save_to_csv("data/flip_threshold.csv"; p_noh = [p_noh], p_h = [p_h])

    # --------------------------------------------------------------#
    # 7)  Print a representative Q‑learning run                     #
    # --------------------------------------------------------------#
    params_demo = ModelParams(w, s̄, k, 0.7, δ, H, L)
    (Q_demo, r_demo, s_demo, _, fired_demo, _, _, _) =
        run_q_learning(params_demo; num_episodes=100, max_steps=ms,
                       alpha_Q=αQ, gamma=γ, temperature=τ0, temp_decay=decay)
    print_qlearning_summary(params_demo, Q_demo, r_demo, s_demo, fired_demo)

    # Print sweep over p
    println("\n================ SUMMARY SWEEP (DP vs RL) =======================")
    for p in pvals
        println("\n----------------  p = $(round(p, digits=2))  -------------------")
        params_sweep = ModelParams(w, s̄, k, p, δ, H, L)
        compare_dp_rl(params_sweep; num_runs=15, num_ep=100, max_steps=ms)
    end

    println("demo_run complete – figures saved in plots/, data in CSV files.")


    params = ModelParams(1.0, 0.5, 2.0, 0.95, 0.95, 2.5, 0.5)
    policy_tables = compare_dp_q_policies(params)

    println("DP vs Q-learning tables finished.")
end

end # module MarkovModelQL