% run_nk_irfs.m
% Main script for the New Keynesian cost-push shock simulations.
% It runs the Dynare model, collects the IRFs, plots them, and saves the outputs.

clear;
close all;
clc;

output_dir = 'Outputs';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% ----------------------------------------------------
% Run the Dynare model
% ----------------------------------------------------

dynare nk_costpush_three_regimes.mod noclearall

% ----------------------------------------------------
% Load the impulse responses
% ----------------------------------------------------

% Horizon shown in the figures
plot_horizon = 10;
t = 0:plot_horizon-1;

% Convert model units into percentage points
scale = 100;

% Inflation
pi_b  = scale * oo_.irfs.pi_b_eps_u(1:plot_horizon);
pi_ag = scale * oo_.irfs.pi_ag_eps_u(1:plot_horizon);
pi_ac = scale * oo_.irfs.pi_ac_eps_u(1:plot_horizon);

% Output gap
x_b  = scale * oo_.irfs.x_b_eps_u(1:plot_horizon);
x_ag = scale * oo_.irfs.x_ag_eps_u(1:plot_horizon);
x_ac = scale * oo_.irfs.x_ac_eps_u(1:plot_horizon);

% Nominal interest rate
i_b  = scale * oo_.irfs.i_b_eps_u(1:plot_horizon);
i_ag = scale * oo_.irfs.i_ag_eps_u(1:plot_horizon);
i_ac = scale * oo_.irfs.i_ac_eps_u(1:plot_horizon);

% ----------------------------------------------------
% Basic summary statistics
% ----------------------------------------------------

regime = {'Accommodative'; 'Benchmark'; 'Aggressive'};

peak_inflation = [
    max(pi_ac);
    max(pi_b);
    max(pi_ag)
];

max_output_gap_loss = [
    -min(x_ac);
    -min(x_b);
    -min(x_ag)
];

peak_nominal_rate = [
    max(i_ac);
    max(i_b);
    max(i_ag)
];

cumulative_output_loss = [
    -sum(x_ac(x_ac < 0));
    -sum(x_b(x_b < 0));
    -sum(x_ag(x_ag < 0))
];

irf_summary_table = table( ...
    regime, ...
    peak_inflation, ...
    max_output_gap_loss, ...
    peak_nominal_rate, ...
    cumulative_output_loss, ...
    'VariableNames', { ...
        'Regime', ...
        'PeakInflation', ...
        'MaxOutputGapLoss', ...
        'PeakNominalRate', ...
        'CumulativeOutputLoss' ...
    } ...
);

disp(' ');
disp('Summary table of IRF statistics');
disp(irf_summary_table);

writetable(irf_summary_table, fullfile(output_dir, 'nk_irf_summary_table.csv'));

% ----------------------------------------------------
% Plot setup
% ----------------------------------------------------

% Colors used in the paper
c_ac = '#86C4FF';
c_b  = [0.6328 0.6364 0.6472];
c_ag = '#0A2C4A';

% Figure style
lw = 2.1;
fs_axes = 11;
fs_title = 12;
fs_legend = 11;

figure('Name','New Keynesian IRFs: Cost-push shock', ...
       'Color','w', ...
       'Units','centimeters', ...
       'Position',[2 2 18 14]);

sgtitle('\textbf{Impulse responses to a cost-push shock under alternative Taylor rules}', ...
        'Interpreter','latex', ...
        'FontSize',13);

% ----------------------------------------------------
% Inflation
% ----------------------------------------------------

subplot(3,1,1)

plot(t, pi_ac, '-',  'Color', c_ac, 'LineWidth', lw); hold on;
plot(t, pi_b,  '--', 'Color', c_b,  'LineWidth', lw);
plot(t, pi_ag, '-.', 'Color', c_ag, 'LineWidth', lw);

yline(0, 'k-', 'LineWidth', 0.8);

title('\textbf{Inflation}', ...
      'Interpreter','latex', ...
      'FontSize',fs_title);

ylabel('p.p.', ...
       'Interpreter','latex', ...
       'FontSize',fs_axes);

legend({'Accommodative','Benchmark','Aggressive'}, ...
       'Interpreter','latex', ...
       'Location','northeast', ...
       'Box','on', ...
       'Color','white', ...
       'EdgeColor',[0.65 0.65 0.65], ...
       'LineWidth',0.6, ...
       'FontSize',fs_legend);

set(gca, ...
    'FontSize',fs_axes, ...
    'Box','off', ...
    'TickLabelInterpreter','latex', ...
    'LineWidth',0.8);

grid on;
ax = gca;
ax.GridAlpha = 0.12;
ax.MinorGridAlpha = 0.08;

% ----------------------------------------------------
% Output gap
% ----------------------------------------------------

subplot(3,1,2)

plot(t, x_ac, '-',  'Color', c_ac, 'LineWidth', lw); hold on;
plot(t, x_b,  '--', 'Color', c_b,  'LineWidth', lw);
plot(t, x_ag, '-.', 'Color', c_ag, 'LineWidth', lw);

yline(0, 'k-', 'LineWidth', 0.8);

title('\textbf{Output gap}', ...
      'Interpreter','latex', ...
      'FontSize',fs_title);

ylabel('p.p.', ...
       'Interpreter','latex', ...
       'FontSize',fs_axes);

legend({'Accommodative','Benchmark','Aggressive'}, ...
       'Interpreter','latex', ...
       'Location','southeast', ...
       'Box','on', ...
       'Color','white', ...
       'EdgeColor',[0.65 0.65 0.65], ...
       'LineWidth',0.6, ...
       'FontSize',fs_legend);

set(gca, ...
    'FontSize',fs_axes, ...
    'Box','off', ...
    'TickLabelInterpreter','latex', ...
    'LineWidth',0.8);

grid on;
ax = gca;
ax.GridAlpha = 0.12;
ax.MinorGridAlpha = 0.08;

% ----------------------------------------------------
% Nominal interest rate
% ----------------------------------------------------

subplot(3,1,3)

plot(t, i_ac, '-',  'Color', c_ac, 'LineWidth', lw); hold on;
plot(t, i_b,  '--', 'Color', c_b,  'LineWidth', lw);
plot(t, i_ag, '-.', 'Color', c_ag, 'LineWidth', lw);

yline(0, 'k-', 'LineWidth', 0.8);

title('\textbf{Nominal interest rate}', ...
      'Interpreter','latex', ...
      'FontSize',fs_title);

ylabel('p.p.', ...
       'Interpreter','latex', ...
       'FontSize',fs_axes);

xlabel('Quarters after shock', ...
       'Interpreter','latex', ...
       'FontSize',fs_axes);

legend({'Accommodative','Benchmark','Aggressive'}, ...
       'Interpreter','latex', ...
       'Location','northeast', ...
       'Box','on', ...
       'Color','white', ...
       'EdgeColor',[0.65 0.65 0.65], ...
       'LineWidth',0.6, ...
       'FontSize',fs_legend);

set(gca, ...
    'FontSize',fs_axes, ...
    'Box','off', ...
    'TickLabelInterpreter','latex', ...
    'LineWidth',0.8);

grid on;
ax = gca;
ax.GridAlpha = 0.12;
ax.MinorGridAlpha = 0.08;

% ----------------------------------------------------
% Save everything
% ----------------------------------------------------

set(gcf, 'PaperPositionMode', 'auto');

exportgraphics(gcf, fullfile(output_dir, 'nk_irfs_three_regimes.png'), 'Resolution', 400);
exportgraphics(gcf, fullfile(output_dir, 'nk_irfs_three_regimes.pdf'), 'ContentType','vector');

save(fullfile(output_dir, 'nk_irfs_results.mat'), ...
    't', ...
    'pi_b', 'pi_ag', 'pi_ac', ...
    'x_b', 'x_ag', 'x_ac', ...
    'i_b', 'i_ag', 'i_ac');

disp('Done.');
disp('Figure saved as Outputs/nk_irfs_three_regimes.png');
disp('Vector figure saved as Outputs/nk_irfs_three_regimes.pdf');
disp('Data saved as Outputs/nk_irfs_results.mat');
