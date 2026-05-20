// nk_costpush_three_regimes.mod
// Standard New Keynesian model with a cost-push shock
// Three monetary policy regimes: benchmark, aggressive, accommodative

var
    x_b pi_b i_b
    x_ag pi_ag i_ag
    x_ac pi_ac i_ac
    u;

varexo eps_u;

parameters
    beta sigma varphi theta kappa
    phi_x rho_u
    phi_pi_b phi_pi_ag phi_pi_ac;

// ----------------------------------------------------
// 1. Calibration of the model
// ----------------------------------------------------

beta   = 0.99;
sigma  = 1;
varphi = 1;
theta  = 0.75;

// Calvo-implied slope of the New Keynesian Phillips Curve
kappa = ((1-theta)*(1-beta*theta)/theta) * (sigma + varphi);

// Monetary policy coefficients
phi_pi_b  = 1.50;   // benchmark Taylor rule
phi_pi_ag = 2.00;   // aggressive regime
phi_pi_ac = 1.10;   // accommodative but still determinate
phi_x     = 0.50/4; // quarterly version of Taylor's output-gap coefficient

// Cost-push shock persistence
rho_u = 0.50;

// ----------------------------------------------------
// 2. Model equations
// Variables are expressed as deviations from steady state
// ----------------------------------------------------

model(linear);

    // Benchmark policy regime
    x_b  = x_b(+1) - (1/sigma)*(i_b - pi_b(+1));
    pi_b = beta*pi_b(+1) + kappa*x_b + u;
    i_b  = phi_pi_b*pi_b + phi_x*x_b;

    // Aggressive policy regime
    x_ag  = x_ag(+1) - (1/sigma)*(i_ag - pi_ag(+1));
    pi_ag = beta*pi_ag(+1) + kappa*x_ag + u;
    i_ag  = phi_pi_ag*pi_ag + phi_x*x_ag;

    // Accommodative policy regime
    x_ac  = x_ac(+1) - (1/sigma)*(i_ac - pi_ac(+1));
    pi_ac = beta*pi_ac(+1) + kappa*x_ac + u;
    i_ac  = phi_pi_ac*pi_ac + phi_x*x_ac;

    // Cost-push shock
    u = rho_u*u(-1) + eps_u;

end;

// ----------------------------------------------------
// 3. Steady state
// The model is linearized: steady state is zero
// ----------------------------------------------------

initval;
    x_b = 0;
    pi_b = 0;
    i_b = 0;

    x_ag = 0;
    pi_ag = 0;
    i_ag = 0;

    x_ac = 0;
    pi_ac = 0;
    i_ac = 0;

    u = 0;
    eps_u = 0;
end;

steady;

// Check Blanchard-Kahn determinacy conditions
check;

// ----------------------------------------------------
// 4. Shock calibration
// ----------------------------------------------------

shocks;
    var eps_u; stderr 0.01;
end;

// ----------------------------------------------------
// 5. Simulation
// ----------------------------------------------------

stoch_simul(order=1, irf=40, nograph)
    pi_b x_b i_b
    pi_ag x_ag i_ag
    pi_ac x_ac i_ac;