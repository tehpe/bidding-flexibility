import pyomo.environ as pyo

#%% Constraints
def add_capacity_market_constraints(model):

    def f_rc_k_rule(m, k):
        return sum(
            m.q_rc[k, i] * sum(m.m_rc[k, j] * m.p_rc[j] for j in range(1, i + 1))
            for i in m.N)

    def f_rc_rule(m):
        return sum(m.f_rc_k[k] for k in m.K)

    def capacity_rc_rule(m, k):
        return sum(m.m_rc[k, i] for i in m.N) <= m.m_rc_max

    model.f_rc_k = pyo.Expression(model.K, rule=f_rc_k_rule)   # eq. (1)
    model.f_rc   = pyo.Expression(rule=f_rc_rule)              # eq. (3)
    model.reserve_capacity = pyo.Constraint(model.K, rule=capacity_rc_rule)  # eq. (2)


def add_energy_market_constraints(model):

    def f_re_k_a_rule(m, k):
        return sum( 
            m.p_re[i] * m.m_re[k,i] * m.a[k,i]/3600
            for i in m.M)
    
    def f_re_limit_rule(m, k):
        return m.f_re_k[k] <= m.f_re_k_a[k]
    
    def f_re_rule(m):
        return sum(m.f_re_k[k] for k in m.K)
    
    def capacity_re_rule(m, k):
        return sum(m.m_re[k, i] for i in m.M) <= m.m_re_max

    model.f_re_k_a = pyo.Expression(model.K, rule = f_re_k_a_rule)         # eq. (5)
    model.re_profit_limit = pyo.Constraint(model.K, rule=f_re_limit_rule)  # eq. (6)
    model.f_re     = pyo.Expression(rule = f_re_rule)                      # eq. (7)
    model.energy_capacity = pyo.Constraint(model.K, rule = capacity_re_rule)  # eq. (4)
    

#%% Model
def build_model(config):

    # data
    products = config["products"]

    n_bids   = config["n_bids"] 
    m_bids   = config["m_bids"] 

    max_flex_rc = config["max_flex_rc"]
    max_flex_re = config["max_flex_re"]

    bid_price_rc_data = config["bid_price_rc_data"]
    accept_prob_data = config["accept_prob_data"]

    bid_price_re_data = config["bid_price_re_data"]
    activation_duration_data = config["activation_duration_data"]

    # create model
    model = pyo.ConcreteModel(name="OptimalBidding")

    # sets
    model.K = pyo.Set(initialize=products)
    model.N = pyo.RangeSet(1, n_bids) # RCM bids
    model.M = pyo.RangeSet(1, m_bids) # REM bids

    # RCM params
    model.p_rc = pyo.Param(model.N, initialize=bid_price_rc_data) 
    model.q_rc = pyo.Param(
        model.K,
        model.N,
        initialize=lambda m, k, i: accept_prob_data[k][i],
        within=pyo.NonNegativeReals
    )
    model.m_rc_max = pyo.Param(initialize=max_flex_rc, within=pyo.NonNegativeReals)

    # REM params
    model.p_re = pyo.Param(model.M, initialize=bid_price_re_data) 
    model.m_re_max = pyo.Param(initialize=max_flex_re, within=pyo.NonNegativeReals) 
    model.a = pyo.Param(
        model.K,
        model.M,
        initialize=lambda m, k, i: activation_duration_data[k][i],
        within=pyo.NonNegativeReals
    )

    # vars
    model.m_rc = pyo.Var(model.K, model.N, domain=pyo.NonNegativeReals)
    model.m_re = pyo.Var(model.K, model.M, domain=pyo.NonNegativeReals)
    model.f_re_k = pyo.Var(model.K, domain=pyo.Reals)    # auxiliary

    # constraints
    add_capacity_market_constraints(model)
    add_energy_market_constraints(model)

    # objective
    model.obj = pyo.Objective(expr=model.f_rc + model.f_re, sense=pyo.maximize)

    return model