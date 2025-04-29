######################################################################
# markov-model-final.jl
#
# A Julia script implementing the two-state productivity model and 
# its infinite-horizon dynamic programming (DP) solution
#
#
# This file includes:
#   1. Parameter struct and basic model definitions
#   2. Value iteration for V_E(α) and V_U(α)
#   3. Policy extraction (keep vs. fire, hire vs. remain unemployed)
#   4. A function to run the DP for given parameters
#   5. A main section demonstrating usage and creating figures
#
#   Author: Marcus A. Lisman, Yale University
#   Written for CSEC 491 Senior Thesis
#
######################################################################

module MarkovModelFinal

using Printf
using Plots
using PrettyTables

######################################################################
# 1. Define the parameters and basic model details
######################################################################

"""
    ModelParams

Data structure holding all relevant parameters for the model:

- w::Float64        : the (fixed) wage
- s_bar::Float64    : the severance (bar{s})
- k::Float64        : the worker's firing cost
- p::Float64        : probability that α remains in the same state (H->H or L->L)
- δ::Float64        : discount factor in (0,1)
- H::Float64        : high productivity level
- L::Float64        : low productivity level
"""
struct ModelParams
    w::Float64       # wage
    s_bar::Float64   # severance level
    k::Float64       # worker's firing cost
    p::Float64       # Markov transition "stay" probability
    δ::Float64       # discount factor
    H::Float64       # "high" productivity
    L::Float64       # "low" productivity
end

"""
    ModelState

We consider two 'employment' states for the firm:
- E (employed)
- U (unemployed)

and two productivity states:
- High (H)
- Low  (L)

For the DP solution, we will store two values: V_E(H) and V_E(L) [employed], and 
two values: V_U(H) and V_U(L) [unemployed].
"""
abstract type ModelState end
struct Employed <: ModelState end
struct Unemployed <: ModelState end


"""
    hostility_cost(params::ModelParams) -> Float64

Returns the hostility cost = (k - s_bar), but only if s_bar < k. 
Otherwise, hostility cost is 0.
"""
function hostility_cost(params::ModelParams)
    return max(0, params.k - params.s_bar)
end


######################################################################
# 2. Value Iteration for V_E(α) and V_U(α)
######################################################################

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
    # We track: V_E(H), V_E(L), V_U(H), V_U(L)
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

######################################################################
# 3. Helper function to pretty-print the results
######################################################################

"""
    print_solution(params::ModelParams, solution)

Prints the value function results and corresponding policies in a neat format.
"""
function print_solution(params::ModelParams, solution)
    (VE_H, VE_L, VU_H, VU_L, policyE_H, policyE_L, policyU_H) = solution

    println("--------------------------------------------------------------")
    println(" Model Parameters:")
    println("   w       = ", params.w)
    println("   s_bar   = ", params.s_bar)
    println("   k       = ", params.k)
    println("   p       = ", params.p)
    println("   δ       = ", params.δ)
    println("   H       = ", params.H)
    println("   L       = ", params.L)
    println("   hostility cost = max(0, k - s_bar) = ", hostility_cost(params))
    println()
    println(" DP Solution (Value Function Results):")
    println("   V_E(H) = ", VE_H)
    println("   V_E(L) = ", VE_L)
    println("   V_U(H) = ", VU_H)
    println("   V_U(L) = ", VU_L)
    println()
    println(" Optimal Policy:")
    println("   (E, H) -> ", policyE_H, "  (keep or fire when employed & α=H)")
    println("   (E, L) -> ", policyE_L, "  (keep or fire when employed & α=L)")
    println("   (U, H) -> ", policyU_H, "  (hire or wait when unemployed & α=H)")
    println("   (U, L) -> wait (no profitable hire at L)")
    println("--------------------------------------------------------------")
end


######################################################################
# 4. Run the DP for a range of p-values and plot results
######################################################################

"""
    run_param_sweep_and_plot(; s_bar, k, w, H, L, δ, p_list)

Runs the dynamic program across a list of p-values, storing the resulting 
optimal values of V_E(H) and V_E(L). Then produces a line plot to visualize 
how the value function changes with p.
"""
function run_param_sweep_and_plot(; 
    s_bar::Float64=1.0, 
    k::Float64=2.0, 
    w::Float64=1.0, 
    H::Float64=2.5, 
    L::Float64=0.5, 
    δ::Float64=0.95, 
    p_list::Vector{Float64} = 0.5:0.05:0.99
)
    VEH_vals = Float64[]
    VEL_vals = Float64[]

    for p in p_list
        # Create model parameters for each p
        params = ModelParams(w, s_bar, k, p, δ, H, L)
        sol = value_iteration(params)
        (VE_H, VE_L, VU_H, VU_L, policyE_H, policyE_L, policyU_H) = sol

        push!(VEH_vals, VE_H)
        push!(VEL_vals, VE_L)
    end

    # Now plot V_E(H) and V_E(L) vs. p
    plt = plot(
        p_list, VEH_vals, 
        xlabel="p (prob of staying in the same state)", 
        ylabel="Value Function",
        label="V_E(H)",
        title="V_E(H) and V_E(L) as functions of p",
        legend=true
    )
    plot!(p_list, VEL_vals, label="V_E(L)")

    return plt
end

######################################################################
# 5. Main
######################################################################

if abspath(PROGRAM_FILE) == @__FILE__
    # Example usage: run a single DP solution

    """
    ModelParams

    - w::Float64        : the (fixed) wage
    - s_bar::Float64    : the severance (bar{s})
    - k::Float64        : the worker's firing cost
    - p::Float64        : probability that α remains in the same state (H->H or L->L)
    - δ::Float64        : discount factor in (0,1)
    - H::Float64        : high productivity level
    - L::Float64        : low productivity level
    """
    myparams = ModelParams(
        1.0,   # w
        0.5,   # s_bar
        2.0,   # k
        0.5,   # p --> 0.5 = i.i.d. case
        0.95,  # δ
        2.5,   # H
        0.5    # L
    )

    solution = value_iteration(myparams)
    print_solution(myparams, solution)
    (VE_H, VE_L, VU_H, VU_L, policyE_H, policyE_L, policyU_H) = value_iteration(myparams)


    # Parameter sweep over p
    p_list = 0.5:0.05:1.0
    plt = run_param_sweep_and_plot(
        s_bar=0.5, k=2.0, w=1.0, H=2.5, L=0.5, δ=0.95, 
        p_list=collect(p_list)
    )

    p_list = 0.5:0.05:1.0

    # Example plot
    savefig(plt, "ValueFunction_vs_p.png")
end

end # module MarkovModelFinal