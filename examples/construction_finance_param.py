 ## Construction Finance Parameters
from ORBIT.phases import install


def con_fin_params(bos, turbine_capex, orbit_install_capex, plant_cap, per_kW=True):
    """ 
    Returns construction finance parameters based on ORCA's equations.

    Can specify if want in $, not $/kW.

    Inputs:
    -------
    bos : float ($)
    turbine_capex : float ($)
    orbit_install_capex : float ($)
    plant_cap : float (kW)
    (Optional) per_kW : bool, default = True

    Outputs:
    --------
    dictionary of the following:
        soft_capex
        construction_insurance_capex
        decomissioning_costs
        construction_financing
        procurement_contingency_costs
        install_contingency_costs
        project_completion_capex

    """

    ## LCOE Factors (from ORCA)
    development_factor = 0.04
    proj_mgmt_factor = 0.02
    construction_insurance = 0.01*1.15
    project_completion = 0.01*1.15
    decom_rate = 0.15*1.15
    procurement_contingency = 0.05*1.15
    install_contingency = 0.30*1.15
    interest_during_construction = 0.044
    tax_rate = 0.26
    spend_schedule = {0: 0.4,
                    1: 0.4,
                    2: 0.2,
                    3: 0.0,
                    4: 0.0,
                    5: 0.0
    }

    ### construction_finance_factor
    _check = 0
    construction_finance_factor = 0

    for key, val in spend_schedule.items():
        _check += val
        if _check > 1.:
            raise Exception("Values in spend_schedule must sum to 1.0")

        construction_finance_factor += val *\
            (1 + (1 - tax_rate) * ((1 + interest_during_construction) ** (key + 0.5) - 1))


    ### construction_insurance_capex
    construction_insurance_capex = construction_insurance * (bos + turbine_capex)

    ###project_completion_capex
    project_completion_capex = project_completion * (bos + turbine_capex)

    ### decomissioning_costs
    decomissioning_costs = decom_rate * orbit_install_capex

    ### Procurement contingency capex
    procurement_contingency_costs = procurement_contingency * (bos - orbit_install_capex + turbine_capex)

    ### install_contingency_costs
    install_contingency_costs = install_contingency * orbit_install_capex


    ### Construction Financing
    tmp1 = construction_insurance_capex +\
        decomissioning_costs +\
        project_completion_capex +\
        procurement_contingency_costs +\
        install_contingency_costs +\
        bos + turbine_capex
    construction_financing = (construction_finance_factor - 1) * tmp1


    ### Soft Capex
    tmp = construction_insurance_capex +\
        decomissioning_costs +\
        project_completion_capex +\
        procurement_contingency_costs +\
        install_contingency_costs
    soft_capex = construction_financing + tmp

    if per_kW:
        soft_capex /= plant_cap
        construction_insurance_capex /= plant_cap
        decomissioning_costs /= plant_cap
        construction_financing /= plant_cap
        procurement_contingency_costs /= plant_cap
        install_contingency_costs /= plant_cap
        project_completion_capex /= plant_cap

    d = {}
    d["soft_capex"] = soft_capex
    d["construction_insurance_capex"] = construction_insurance_capex
    d["decomissioning_costs"] = decomissioning_costs
    d["construction_financing"] = construction_financing
    d["procurement_contingency_costs"] = procurement_contingency_costs
    d["install_contingency_costs"] = install_contingency_costs
    d["project_completion_capex"] = project_completion_capex
    return d


