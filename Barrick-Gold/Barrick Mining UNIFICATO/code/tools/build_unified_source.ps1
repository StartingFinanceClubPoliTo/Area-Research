param(
    [string]$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
)

$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath (Join-Path $WorkspaceRoot 'Utilities-Progetto\REFRESH-PLAN-2026-09-05.md')) {
    throw 'Legacy August generator: blocked to preserve the September editorial refresh. Use the current Overleaf source; reconstruct historical material only in a separate archived workspace.'
}
$projectRoot = (Resolve-Path (Join-Path $WorkspaceRoot '..')).Path
$chapterRoot = Join-Path $WorkspaceRoot 'Overleaf\chapters'
New-Item -ItemType Directory -Force $chapterRoot | Out-Null

function Get-SourceSlice {
    param(
        [string]$RelativePath,
        [int]$Start,
        [int]$End
    )

    $sourcePath = Join-Path $projectRoot $RelativePath
    $lines = Get-Content -Encoding UTF8 -LiteralPath $sourcePath
    if ($Start -lt 1 -or $End -gt $lines.Count -or $Start -gt $End) {
        throw "Invalid slice ${RelativePath}:${Start}-${End} (line count $($lines.Count))."
    }
    return ($lines[($Start - 1)..($End - 1)] -join "`n")
}

function Convert-ToUnifiedFragment {
    param([string]$Text)

    # Floats and embedded code are deliberately excluded until their dedicated
    # provenance/recompute ledgers accept them. Equations and prose are retained.
    foreach ($environment in @('figure\*?', 'table\*?', 'longtable', 'tikzpicture', 'lstlisting', 'minted', 'verbatim', 'pseudocode')) {
        $pattern = "(?ms)\\begin\{$environment\}.*?\\end\{$environment\}"
        $Text = [regex]::Replace($Text, $pattern, "% Asset/code block omitted pending provenance acceptance.`n")
    }
    $Text = [regex]::Replace($Text, '(?m)^\s*\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}\s*$', '% Source image omitted pending provenance acceptance.')

    # Citation and cross-reference keys remain recoverable in the immutable
    # source. The master renders explicit audit markers instead of creating
    # misleading unresolved LaTeX links after figures/tables were withheld.
    $Text = [regex]::Replace($Text, '\\(?:cite|citep|citet)(?:\[[^\]]*\]){0,2}\{([^}]*)\}', '\sourcecitation{$1}')
    $Text = [regex]::Replace($Text, '\\(?:eqref|ref|autoref)\{([^}]*)\}', '\sourceref{$1}')
    $Text = [regex]::Replace($Text, '\\label\{[^}]*\}', '')
    $displayEnvironment = '(?ms)\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}(.*?)\\end\{\1\}'
    $Text = [regex]::Replace($Text, $displayEnvironment, {
        param($match)
        return [regex]::Replace($match.Value, '(?m)^\s*$\n', '')
    })
    $Text = [regex]::Replace($Text, '(?ms)\\\[(.*?)\\\]', {
        param($match)
        return [regex]::Replace($match.Value, '(?m)^\s*$\n', '')
    })

    # Break the Hull--Dobell conditions over three lines; the legacy source
    # placed all three clauses in one display wider than the Research template.
    $longPeriodConditions = @'
c \text{ and } m \text{ are coprime}, \qquad
a - 1 \text{ is a multiple of all prime factors of } m, \qquad
a - 1 \text{ is a multiple of } 4 \text{ if } m \text{ is a multiple of } 4,
'@
    $wrappedPeriodConditions = @'
\begin{gathered}
c \text{ and } m \text{ are coprime},\\
a - 1 \text{ is a multiple of all prime factors of } m,\\
a - 1 \text{ is a multiple of } 4 \text{ if } m \text{ is a multiple of } 4.
\end{gathered}
'@
    $Text = $Text.Replace($longPeriodConditions.Trim(), $wrappedPeriodConditions.Trim())

    # Nest each legacy article below the canonical unified chapter. Convert
    # deepest-to-shallowest: the previous shallow-to-deep order re-matched its
    # own replacements and collapsed every heading to \paragraph, leaving
    # 20-page stretches without section/subsection entries in the TOC.
    $Text = [regex]::Replace($Text, '(?m)^\s*\\subsubsection\*?\{', '\paragraph{')
    $Text = [regex]::Replace($Text, '(?m)^\s*\\subsection\*?\{', '\subsubsection{')
    $Text = [regex]::Replace($Text, '(?m)^\s*\\section\*?\{', '\subsection{')
    $Text = [regex]::Replace($Text, '(?m)^\s*\\chapter\*?\{', '\section{')

    # Keep mathematical heading text readable in print while supplying plain
    # PDF-bookmark strings to hyperref.
    $Text = $Text.Replace(
        '\subsection{Testing Hypotheses about a Single Parameter: The $t$-Test}',
        '\subsection{Testing Hypotheses about a Single Parameter: The \texorpdfstring{$t$}{t}-Test}'
    )
    $Text = $Text.Replace(
        '\subsection{Volatility Scaling ($\beta$)}',
        '\subsection{Volatility Scaling (\texorpdfstring{$\beta$}{beta})}'
    )
    $Text = [regex]::Replace(
        $Text,
        '(?m)^\\subsection\{Example: Monte Carlo Estimation of the Integral .*$',
        { param($match) return '\subsection{Example: Monte Carlo Estimation of the Integral \texorpdfstring{$\int_0^1 e^{-x^2}\,dx$}{integral from 0 to 1 of exp(-x squared) dx}}' }
    )

    # The source files occasionally contain page-layout commands that would
    # interfere with the canonical template when moved into a master chapter.
    $Text = [regex]::Replace($Text, '(?m)^\s*\\(?:clearpage|newpage|pagestyle|thispagestyle)\b.*$', '')
    return $Text.Trim()
}

$specs = @(
    [pscustomobject]@{
        File = '04-probability-and-pricing.tex'
        Title = 'Probability and Stochastic-Pricing Foundations'
        Status = 'Edited reuse from Team 1. Mathematical and source-to-claim review remains open; figures are withheld pending the figure ledger.'
        Sources = @(
            [pscustomobject]@{ Path = 'Team 1 - Black-Scholes Foundations\Overleaf\Articolo.tex'; Start = 159; End = 1368 }
        )
    },
    [pscustomobject]@{
        File = '05-econometrics-and-time-series.tex'
        Title = 'Econometrics, Time Series and Dynamic Dependence'
        Status = 'Edited reuse from Team 2. General theory is retained; application-specific claims require the unified validation protocol.'
        Sources = @(
            [pscustomobject]@{ Path = 'Team 2 - OLS Dynamic Correlations\Overleaf\Articolo.tex'; Start = 209; End = 630 },
            [pscustomobject]@{ Path = 'Team 2 - OLS Dynamic Correlations\Overleaf\Articolo.tex'; Start = 876; End = 1255 }
        )
    },
    [pscustomobject]@{
        File = '06-monte-carlo.tex'
        Title = 'Monte Carlo Methods and Risk Simulation'
        Status = 'Edited reuse from Team 3. Numerical outputs and source figures are excluded until parity and recomputation gates are closed.'
        Sources = @(
            [pscustomobject]@{ Path = 'Team 3 - Monte Carlo Risk\Overleaf\Articolo.tex'; Start = 198; End = 460 },
            [pscustomobject]@{ Path = 'Team 3 - Monte Carlo Risk\Overleaf\Articolo.tex'; Start = 461; End = 1076 },
            [pscustomobject]@{ Path = 'Team 3 - Monte Carlo Risk\Overleaf\Articolo.tex'; Start = 1077; End = 1682 },
            [pscustomobject]@{ Path = 'Team 3 - Monte Carlo Risk\Overleaf\Articolo.tex'; Start = 1683; End = 1877 },
            [pscustomobject]@{ Path = 'Team 3 - Monte Carlo Risk\Overleaf\Articolo.tex'; Start = 2234; End = 2578 }
        )
    },
    [pscustomobject]@{
        File = '07-gold-asset.tex'
        Title = 'Gold as a Monetary and Financial Asset'
        Status = 'Edited reuse from Team 7. Historical Yahoo-based evidence remains a legacy sample and is not merged with the LSE option sample.'
        Sources = @(
            [pscustomobject]@{ Path = 'Team 7 - Gold Volatility Dynamics\Overleaf\Articolo.tex'; Start = 244; End = 767 }
        )
    },
    [pscustomobject]@{
        File = '08-volatility-jumps-affine.tex'
        Title = 'Volatility, Jumps and Affine Stochastic Models'
        Status = 'Edited reuse from Team 6. Synthetic illustrations and calibration claims are excluded; formula parity and branch conventions remain under review.'
        Sources = @(
            [pscustomobject]@{ Path = 'Team 6 - Beyond Black-Scholes\Overleaf\Articolo.tex'; Start = 406; End = 500 },
            [pscustomobject]@{ Path = 'Team 6 - Beyond Black-Scholes\Overleaf\Articolo.tex'; Start = 501; End = 894 },
            [pscustomobject]@{ Path = 'Team 6 - Beyond Black-Scholes\Overleaf\Articolo.tex'; Start = 895; End = 1588 },
            [pscustomobject]@{ Path = 'Team 6 - Beyond Black-Scholes\Overleaf\Articolo.tex'; Start = 1589; End = 2258 },
            [pscustomobject]@{ Path = 'Team 6 - Beyond Black-Scholes\Overleaf\Articolo.tex'; Start = 2428; End = 2650 }
        )
    },
    [pscustomobject]@{
        File = '09-option-calibration.tex'
        Title = 'Calibration, Model Comparison and Gold-Price Simulation'
        Status = 'Edited reuse from Team 8 plus accepted frozen and current diagnostics. The fitted object is a GLD option surface under the risk-neutral measure; simulation and corporate valuation remain separately identified interfaces.'
        Sources = @(
            [pscustomobject]@{ Path = 'Team 8 - Gold Options Stochastic Modeling\Overleaf\Articolo.tex'; Start = 238; End = 1544 }
        )
    },
    [pscustomobject]@{
        File = '10-operations-ebitda.tex'
        Title = 'Production, Costs and Stochastic EBITDA'
        Status = 'Edited reuse from Team 4. The operating baseline exists, while GLD-to-gold basis, measure semantics and current Barrick inputs remain open.'
        Sources = @(
            [pscustomobject]@{ Path = 'Team 4 - Component-Driven EBITDA\Overleaf\Articolo.tex'; Start = 164; End = 866 }
        )
    }
)

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
foreach ($spec in $specs) {
    $parts = foreach ($source in $spec.Sources) {
        Convert-ToUnifiedFragment (Get-SourceSlice -RelativePath $source.Path -Start $source.Start -End $source.End)
    }
    $body = $parts -join "`n`n% ---- next approved source slice ----`n`n"

    # The Team 8 source freeze contains an accepted OOS table and a historical
    # cumulative-loss figure.  The generic converter removes every float.  The
    # unified project instead reinstates the accepted table and the dedicated
    # publication chart rebuilt from byte-frozen aggregate metrics after
    # identity/sample/date validation (CODE-009).  Keep this post-processing
    # here so a later source rebuild cannot silently erase the accepted asset.
    if ($spec.File -eq '09-option-calibration.tex') {
        $oosTable = @'
\begin{table}[H]
\centering
\scriptsize
\setlength{\tabcolsep}{3.4pt}
\begin{tabular}{@{}lrrrr@{}}
\toprule
Model & MAE bp & RMSE bp & \(R^2_{OOS}\) vs mean &
\(R^2_{OOS}\) vs previous-day IV \\
\midrule
Black--Scholes & 111.71 & 148.09 & 91.33\% & $-3.76$\% \\
Heston & 122.99 & 165.93 & 89.12\% & $-30.26$\% \\
Bates & 116.62 & 159.85 & 89.90\% & $-20.88$\% \\
Full Bates--Hawkes & 110.90 & 150.96 & 91.00\% & $-7.82$\% \\
\bottomrule
\end{tabular}
\caption{Rolling one-step-ahead implied-volatility forecasts on 3,052 common
node-date observations and 124 target dates from 11 February to 10 August
2026. The previous-day benchmark is the observed normalized IV at the same
fixed node. Positive \(R^2_{OOS}\) favors the model; negative values imply that
carrying forward the previous surface has lower aggregate squared error.}
\label{tab:online-oos}
\end{table}
'@
        $tablePattern = '(?ms)(A Newey--West HAC test.*?zero \\sourcecitation\{NeweyWest1987\}\.)\s*% Asset/code block omitted pending provenance acceptance\.\s*'
        if (-not [regex]::IsMatch($body, $tablePattern)) {
            throw 'Team 8 OOS table insertion anchor not found.'
        }
        $body = [regex]::Replace($body, $tablePattern, {
            param($match)
            return $match.Groups[1].Value + "`n`n" + $oosTable.Trim() + "`n`n"
        }, 1)

        $oosFigure = @'
\IfFileExists{img/diagnostics/online_oos_r2_by_benchmark.png}{%
\begin{figure}[H]
    \centering
    \includegraphics[width=0.98\textwidth]{img/diagnostics/online_oos_r2_by_benchmark.png}
    \caption{Team 8 rolling one-step-ahead out-of-sample \(R^2\) by model and
    benchmark. All four specifications improve on the node-specific expanding
    mean, but none improves on carrying forward the previous day's normalized
    IV surface. The chart is rebuilt offline from the byte-frozen aggregate;
    it does not claim independent historical market-parity replication.}
    \label{fig:online-oos-r2}
\end{figure}
}{%
\gapbox{The Team 8 OOS \(R^2\) chart is withheld unless it is regenerated from
the frozen online-validation metrics and passes the sample and identity checks.}%
}
'@
        $figurePattern = '(?ms)\\IfFileExists\{img/diagnostics/online_welch_goyal_cumulative\.png\}\{%\s*% Asset/code block omitted pending provenance acceptance\.\s*\}\{\}'
        if (-not [regex]::IsMatch($body, $figurePattern)) {
            throw 'Team 8 OOS figure insertion anchor not found.'
        }
        $body = [regex]::Replace($body, $figurePattern, $oosFigure.Trim(), 1)

        # CODE-012/FIG-006 accepts one regenerated four-model path-band figure.
        # It replaces the omitted Team 8 legacy summary and must survive later
        # source rebuilds. The caption keeps the GLD/Q, Q-to-P and Team 4/5
        # boundaries visible.
        $goldBandsFigure = @'
\IfFileExists{img/valuation/gold_path_bands_four_models.png}{%
\begin{figure}[H]
    \centering
    \includegraphics[width=0.98\textwidth]{img/valuation/gold_path_bands_four_models.png}
    \caption{Conditional P10--P50--P90 gold-path bands for Black--Scholes/GBM,
    Heston, Bates--Poisson and Full Bates--Hawkes. The 8,192 paths share the
    Team 4 start of USD 4,677/troy oz, a five-year/20-quarter horizon and
    aligned base seeds. The option-implied shape comes from the frozen
    12 August 2026 Team 8 calibration; the 26 August LSE surface is an audit
    only. These are risk-neutral conditional scenarios, not a validated
    physical forecast or a Q-to-P mapping.}
    \label{fig:gold-path-bands-four-models}
\end{figure}
}{%
\gapbox{The four-model gold-path bands are withheld unless the accepted
CODE-012 figure bundle and its CSV/JSON sidecars are available.}%
}
'@
        $goldBandsPattern = '(?ms)\\IfFileExists\{img/diagnostics/gold_path_stats_by_model\.png\}\{%\s*% Asset/code block omitted pending provenance acceptance\.\s*\}\{\}'
        if (-not [regex]::IsMatch($body, $goldBandsPattern)) {
            throw 'Team 8 gold-path band insertion anchor not found.'
        }
        $body = [regex]::Replace($body, $goldBandsPattern, $goldBandsFigure.Trim(), 1)

        $body = $body.Replace(
            'These numbers explain the current-snapshot modelling choice.',
            "These numbers explain the Team 8 option-layer choice and feed the controlled`nfour-model Barrick comparison in Chapter~12. They do not by themselves select`na company value or validate the commodity-to-corporate bridge."
        )
        $body = $body.Replace(
            "This stress result measures six-month parameter stability; it never replaces the current LSE calibration used for today's Barrick/GLD scenario layer.",
            'This stress result measures six-month parameter stability; it does not replace the current-snapshot calibration used by the controlled Barrick comparison.'
        )
        $body = $body.Replace(
            'Earlier company-level modelling also shows why a future Barrick framework will need gold-price scenarios, but the present article stops at the construction and testing of the gold-price process \sourcecitation{BarrickArticle4,BarrickArticle5}.',
            "Earlier company-level modelling also shows why a Barrick framework needs gold-price scenarios. This Team 8 module stops at construction and testing of the price process; Chapter~12 performs the separate corporate integration \sourcecitation{BarrickArticle4,BarrickArticle5}."
        )
        $body = $body.Replace(
            'The present article supplies the price-model block only.',
            'The present chapter supplies the price-model block only.'
        )
        $body = $body.Replace(
            '\subsection{Boundary with the Future Unified Project}',
            '\subsection{Interface with the Unified Barrick Experiment}'
        )
        $body = $body.Replace(
            'The practical reason for introducing Heston, Bates, and the Bates--Hawkes extension is to prepare a better gold-price input for later work. This article does not estimate Barrick revenues, EBITDA, cash flows, or equity value. Its output is narrower: calibrated parameters, option-surface diagnostics, and simulated gold-price paths.',
            "The practical reason for introducing Heston, Bates, and the Bates--Hawkes extension is to provide a stronger gold-price input. This Team 8 module does not estimate Barrick revenues, EBITDA, cash flows or equity value by itself. Its output is narrower: calibrated parameters, option-surface diagnostics and simulated gold-price paths. Chapter~12 connects that output to the separate Team 4 operating layer and Team 5 corporate valuation contract."
        )
        $body = $body.Replace(
            'The future unified project can then decide how to connect those paths to production, costs, and company-level valuation assumptions. For that reason, the price simulator should expose a clean interface:',
            'The unified experiment connects those paths to production, costs and company-level valuation assumptions through a clean interface:'
        )
        $body = $body.Replace(
            'where \(S_t^{(m)}\) is the simulated gold price, \(v_t^{(m)}\) is the variance path when the model has stochastic volatility, \(N_t^{(m)}\) is the jump count, and \(\lambda_t^{(m)}\) is the jump intensity for the Hawkes extension. The superscript \(m\) denotes the Monte Carlo path. Keeping this boundary explicit prevents the article from mixing two different tasks: modelling the gold price today and modelling corporate revenues in a later unified article.',
            'where \(S_t^{(m)}\) is the simulated gold price, \(v_t^{(m)}\) is the variance path when the model has stochastic volatility, \(N_t^{(m)}\) is the jump count, and \(\lambda_t^{(m)}\) is the jump intensity for the Hawkes extension. The superscript \(m\) denotes the Monte Carlo path. Keeping this boundary explicit prevents the article from confusing two different tasks: modelling the gold price and mapping that price into corporate revenues and value.'
        )
        $body = $body.Replace(
            'The models in this article are research tools.',
            'The models in this chapter are research tools.'
        )
        $body = $body.Replace(
            'Self-excitation provides the lowest current cross-sectional error and a coherent clustered-jump simulator. It does not establish out-of-sample dominance from one option snapshot, and it does not make residuals Gaussian. A dated panel of option surfaces would be required to test parameter stability and a time-varying clustering premium. The following section therefore compares all four models and treats Monte Carlo outputs as conditional scenarios, not forecasts or recommendations.',
            'Self-excitation provides the lowest current cross-sectional error and a coherent clustered-jump simulator. It does not establish predictive dominance and it does not make residuals Gaussian. The dated rolling panel reported below performs the parameter-stability and one-step-ahead test that the standalone article originally left as a next step: it shows that no candidate beats the previous-day IV surface. Monte Carlo outputs are therefore treated as conditional scenarios, not forecasts or recommendations.'
        )
        $body = $body.Replace(
            'The target implementation should first test whether a single Sobol-based engine with moment matching, common random numbers, and model-specific transformations is robust enough for all path generators. If that unified design is too rigid, the code should expose model-specific policies. A GBM path only needs Brownian shocks, while a Bates--Hawkes path needs diffusive shocks, stochastic variance, jump marks, and a self-exciting arrival mechanism. Treating these models as if they had the same random structure would make the path simulation mechanically tidy but financially weak.',
            'The accepted implementation uses one Sobol-based simulation interface with moment matching, aligned base seeds and model-specific transformations, while exposing separate policies for diffusion and event-driven blocks. A GBM path needs one Brownian shock, whereas a Bates--Hawkes path needs diffusive shocks, stochastic variance, jump marks and a self-exciting arrival mechanism. The completed design therefore shares what is comparable without pretending that all four models have the same random structure.'
        )
        $body = $body.Replace(
            'The equations below define the model updates. The sampling policy should be passed as a simulation configuration, not hidden inside the financial model itself.',
            'The equations below define the implemented model updates. The sampling policy is passed as a simulation configuration rather than hidden inside the financial model itself.'
        )
        $body = $body.Replace(
            'For Heston, the Monte Carlo engine should simulate both price and variance. A full-truncation Euler scheme can be written as',
            'For Heston, the Monte Carlo engine simulates both price and variance. Its full-truncation Euler scheme is'
        )
        $body = $body.Replace(
            'With the simulation boundary and the non-recommendation scope fixed, the final chapter can summarize what the calibrated price layer contributes to the Barrick research track.',
            'With the simulation boundary and the non-recommendation scope fixed, the unified valuation chapter can use the calibrated price layer without repeating its fit and predictive tests.'
        )
        $body = $body.Replace(
            "\end{itemize}`n`nThis is why the article emphasizes diagnostics rather than only model formulas.",
            "\end{itemize}`n`n\input{chapters/09-team8-calibration-evidence}`n`nThis is why the article emphasizes diagnostics rather than only model formulas."
        )
        $body = $body.Replace(
            'Keeping volatility, Hawkes jumps, and Bates jumps in three figures avoids compressing a nearly constant intensity beside a much more variable volatility process.',
            "Keeping volatility, Hawkes jumps, and Bates jumps in three figures avoids compressing a nearly constant intensity beside a much more variable volatility process.`n`n\input{chapters/09-team8-simulation-evidence}"
        )
    }
    if ($spec.File -eq '10-operations-ebitda.tex') {
        $body = $body.Replace(
            "Because Barrick is a massive, globally diversified company, these localized risks rarely happen all at once. These operational shocks are isolated to a specific mine or region, rather than being systemic. Therefore, we believe that the probability of simultaneous failure is statistically negligible. In the production forecast, we relied on the fact that the shocks would be absorbed, both geographically and over time.`n\section{EBITDA Forecast Methodology}",
            "Because Barrick is a massive, globally diversified company, these localized risks rarely happen all at once. These operational shocks are isolated to a specific mine or region, rather than being systemic. Therefore, we believe that the probability of simultaneous failure is statistically negligible. In the production forecast, we relied on the fact that the shocks would be absorbed, both geographically and over time.`n\input{chapters/10-team4-operations-evidence}`n\section{EBITDA Forecast Methodology}"
        )
        $body = $body.Replace(
            'A negative lower bound emerges in the final year of the simulation.',
            "\input{chapters/10-team4-ebitda-evidence}`n`nA negative lower bound emerges in the final year of the simulation."
        )
        $body = $body.Replace(
            'A more refined formulation incorporating these production-side real options is left to future work and discussed briefly in Section~\sourceref{sec:limitations}.',
            'The resulting production-side limitation is separated from the price-engine and corporate-integration steps already completed by the unified project.'
        )
        $body = $body.Replace(
            '\subsection{Limitations and Next Steps}',
            "\subsection{Limitations and Downstream Resolution}\label{sec:limitations}`n`nThe standalone Team 4 article treated a richer gold-price law and connection to corporate valuation as subsequent work. Those two steps are now executed: Chapter~9 supplies the four calibrated price engines and Chapter~12 propagates each one through a common operating and DCF contract. They are therefore reported as completed downstream results, not as recommendations for future work. What remains open on the operating side is economically different: the frozen production and cost vectors do not react endogenously to adverse price states, mine depletion, grade changes or managerial real options. These limitations enter the present validation boundary and the genuinely new research programme stated in the unnumbered conclusion."
        )
    }
    $header = @"
% AUTO-GENERATED by Github-Branch/tools/build_unified_source.ps1.
% Do not edit this file directly: update the transformation manifest instead.
\chapter{$($spec.Title)}
\chapterstatus{$($spec.Status)}
\sourceassetnote

"@
    [System.IO.File]::WriteAllText((Join-Path $chapterRoot $spec.File), $header + $body + "`n", $utf8NoBom)
}

Write-Output "Generated $($specs.Count) provenance-traced chapter fragments in $chapterRoot."
