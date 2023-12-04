# AEP Calculation Script
# Patrick Duffy
# NREL 2023
# See https://www.nrel.gov/docs/fy21osti/77384.pdf details


################################ INPUTS #######################################
wake_loss_fixed = .05        # <------ UPDATE with FLORIS result
wake_loss_floating = .05     # <------ UPDATE with FLORIS result
avail_loss_fixed = .05      # <------ UPDATE with WOMBAT result
avail_loss_floating = .05   # <------ UPDATE with WOMBAT result
depth_fixed = 34            # COE Review floating site depth (m)
depth_floating = 739        # COE Review floating site depth (m)
export_fixed = 50           # COE Review floating site cable length (km)
export_floating = 36        # COE Review floating site cable length (km)
turb_capacity = 12          # COE Review (MW)
num_turbines = 84           # COE Review 

############################### CONSTANTS #####################################
enviro_loss = 0.0159
tech_loss_fixed = .01 # Hysterisis (1%)
tech_loss_floating = 1 - (1 - 0.01) * (1 - 0.001) * (1 - 0.001) # Hysteresis (1%), Onboard Equip. (0.1%), Rotor Misalignment (0.1%)


################################# METHODS #####################################
def elec_loss(depth, dist_s_to_l):
    """
    Calculates and returns electrical losses. Comes from ORCA model in Beiter 
    et al. (2016).

    Parameters
    ----------
    data : Data
        Contains geographic data for the site(s).

    Returns
    -------
    elec_loss : array
    """

    elec_loss = (2.20224112 +
                    0.000604121 * depth +
                    0.0407303367321603 * dist_s_to_l +
                    -0.0003712532582 * dist_s_to_l ** 2 +
                    0.0000016525338 * dist_s_to_l ** 3 +
                    -0.000000003547544 * dist_s_to_l ** 4 +
                    0.0000000000029271 * dist_s_to_l ** 5) / 100
    
    return elec_loss


def total_loss(enviro_loss, tech_loss, wake_loss, elec_loss, avail_loss):
    site_spec_loss = (1 - ((1 - wake_loss) * (1 - elec_loss) * (1 - avail_loss)))
    total_loss = (1 - (1 - enviro_loss) * (1 - tech_loss) * (1 - site_spec_loss))

    return total_loss

################################ CALCULATE ####################################
elec_loss_fixed = elec_loss(depth_fixed, export_fixed)
elec_loss_floating = elec_loss(depth_floating, export_floating)

total_loss_fixed = total_loss(enviro_loss, 
                                tech_loss_fixed, 
                                wake_loss_fixed, 
                                elec_loss_fixed, 
                                avail_loss_fixed)

total_loss_floating = total_loss(enviro_loss, 
                                tech_loss_floating, 
                                wake_loss_floating, 
                                elec_loss_floating, 
                                avail_loss_floating)

print("Fixed total loss: ", total_loss_fixed)
print("Floating total loss: ", total_loss_floating)

# Compare the results to the 2021 COE Review:
# Fixed: 58% GCF, 15.5% total loss, 49% NCF, 4,295 MWh/MW/yr net energy capture
# Floating: 48% GCF, 20.7% total loss, 38.1% NCF, 3,336 MWh/MW/yr net energy capture

# If our results are dramatically different, let's tune the "additional losses" category
# in WAVES after getting wake and avail losses so we match our old total loss 

# my goal here is to be more transparent, but DOE wants to review every time
# we make some tweaks

##############################################################################
# Optional section - depends on what your outputs from FLORIS are
# I would like to report GCF, all losses, NCF 
# as well as Net AEP in terms of MWh/MW/yr since that is the format in COE Review

# gcf = .5 #random choice, fill in
# ncf = gcf * (1 - total_loss)
# gross_aep = gcf * turb_capacity * num_turbines * 8760
# net_aep = ncf * turb_capacity * num_turbines * 8760
