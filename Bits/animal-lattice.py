"""
Animal Lattice — Emergent Connectivity Layer
=============================================
Eight animal-derived network geometries as basis functions
for biological connectivity. Each represents a distinct
topology observed in animal cognition, communication,
or movement.

Geometries:
    Raven     — small-world graph optimization
    Octopus   — combinatorial hypercube mapping
    Dolphin   — echoic harmonic resonance
    Parrot    — vocal mimicry / adaptive copying
    Spider    — web tension matrix
    Bee       — swarm dynamics
    Elephant  — memory-weighted edges
    Wolf      — pack coordination

Phi scaling: 1, 1.618, 2.618, 4.236, 6.854, 11.090
Spin states: up (reinforcing) / down (damping)

Control panel coordinates from cross-model synthesis:
    [0.5927, 60]      — baseline connectivity, t_c
    [38, 89]          — C_seed, asymptotic target
    [3.1415926, 42]   — phase factors, angular frequencies
    [0.139, 0.875]    — critical exponents

This file is a basis function library.
Import into integrative_sim_v2.py via compute_C_new(t, state).
"""

import numpy as np

# ── Constants ──────────────────────────────────────────────
PHI = 1.6180339887
PHI_SERIES = [PHI**n for n in range(6)]
# [1.0, 1.618, 2.618, 4.236, 6.854, 11.090]

# Control panel
BASELINE_AC = 0.5927
T_C         = 60
C_SEED      = 38 / 89        # ratio: seed to asymptotic
BETA_TARGET = 89
PI_PHASE    = 3.1415926
OMEGA       = 42
ALPHA_OLD   = 0.139
ALPHA_NEW   = 0.875


# ── Utility ────────────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normalize(x, eps=1e-10):
    r = np.max(x) - np.min(x)
    if r < eps:
        return np.zeros_like(x)
    return (x - np.min(x)) / r

def phi_scale(value, n):
    """Scale value by nth power of phi."""
    return value * PHI_SERIES[n]


# ── Eight geometry functions ───────────────────────────────

def G_raven(t, n_nodes=20, rewire_prob=0.15):
    """
    Raven — Graph Optimization
    Small-world network. High local clustering plus
    long-range shortcuts. Models raven social networks:
    dense local groups with occasional long-distance
    individual connections that bridge clusters.

    Spin: UP — global shortcuts increase connectivity
    Phi note: 1 (base, foundational)

    Returns scalar connectivity value in [0,1].
    """
    np.random.seed(int(t * 7) % 10000)

    # Ring lattice base
    adj = np.zeros((n_nodes, n_nodes))
    k = 4  # each node connected to k nearest neighbors
    for i in range(n_nodes):
        for j in range(1, k//2 + 1):
            adj[i, (i+j) % n_nodes] = 1
            adj[i, (i-j) % n_nodes] = 1

    # Rewire with probability rewire_prob (Watts-Strogatz)
    for i in range(n_nodes):
        for j in range(1, k//2 + 1):
            if np.random.random() < rewire_prob:
                new_j = np.random.randint(0, n_nodes)
                if new_j != i and adj[i, new_j] == 0:
                    adj[i, (i+j) % n_nodes] = 0
                    adj[i, new_j] = 1

    # Connectivity metric: mean degree normalized
    mean_degree = adj.sum() / n_nodes
    G = mean_degree / (n_nodes - 1)

    # Phi-scaled growth with time
    growth = 1 - np.exp(-phi_scale(0.05, 0) * t)
    return float(np.clip(G * growth, 0, 1))


def G_octopus(t, dims=4):
    """
    Octopus — Combinatorial Hypercube Mapping
    Hypercube topology. 2^dims nodes each connected to
    dims neighbors differing by one bit. Models octopus
    distributed neural processing: arms operate
    semi-independently, high-dimensional local mapping.

    Spin: UP — combinatorial richness increases with dims
    Phi note: A4 (2.618)

    Returns scalar connectivity value in [0,1].
    """
    n_nodes = 2 ** dims
    edges = n_nodes * dims / 2  # each node has dims edges

    # Maximum possible edges
    max_edges = n_nodes * (n_nodes - 1) / 2

    # Base connectivity from hypercube structure
    base_G = edges / max_edges

    # Octopus adapts rapidly — connectivity peaks then stabilizes
    # Models rapid camouflage/problem-solving then consolidation
    adaptation = np.tanh(phi_scale(0.08, 2) * t)
    decay      = np.exp(-0.002 * t)

    G = base_G * adaptation * (0.7 + 0.3 * decay)
    return float(np.clip(G, 0, 1))


def G_dolphin(t, omega=OMEGA, phase=PI_PHASE):
    """
    Dolphin — Echoic Harmonic Resonance
    Oscillating connectivity. Models dolphin echolocation
    and social acoustic networks: periodic strengthening
    and weakening of connections as acoustic windows open
    and close. Harmonic structure at phi-scaled frequencies.

    Spin: DOWN then UP — initial damping then resonance builds
    Phi note: 1.618 (phi itself)

    Returns scalar connectivity value in [0,1].
    """
    # Primary harmonic
    f1 = np.sin(2 * np.pi * t / omega + phase)

    # Phi-scaled harmonics
    f2 = np.sin(2 * np.pi * t / (omega / PHI) + phase) / PHI
    f3 = np.sin(2 * np.pi * t / (omega / PHI**2) + phase) / PHI**2

    # Envelope: grows with time as acoustic network matures
    envelope = sigmoid(0.05 * (t - T_C/3))

    G = envelope * normalize(
        np.array([f1 + f2/PHI + f3/PHI**2])
    )[0]

    # Ensure positive
    G = (G + 1) / 2
    return float(np.clip(G, 0, 1))


def G_parrot(t, history=None):
    """
    Parrot — Vocal Mimicry / Adaptive Copying
    History-dependent connectivity. Models parrot
    vocal learning: connectivity at time t is weighted
    by what has been 'heard' (experienced) previously.
    Strong signals get amplified. Rare signals fade.

    Spin: UP — reinforces successful pathways
    Phi note: D5 (4.236)

    history: list of past G values (any geometry)
    Returns scalar connectivity value in [0,1].
    """
    if history is None or len(history) == 0:
        # No history: start with baseline
        return float(sigmoid(0.01 * t - 2))

    history = np.array(history)

    # Exponential recency weighting
    weights = np.exp(
        phi_scale(0.1, 3) *
        np.linspace(-1, 0, len(history))
    )
    weights /= weights.sum()

    # Weighted mean of history
    G_memory = np.dot(weights, history)

    # Mimicry: amplify strong signals, dampen weak
    if G_memory > 0.5:
        G = G_memory * PHI_SERIES[1] / PHI_SERIES[2]
    else:
        G = G_memory * PHI_SERIES[0] / PHI_SERIES[1]

    return float(np.clip(G, 0, 1))


def G_spider(t, n_radial=8, n_spiral=12):
    """
    Spider — Web Tension Matrix
    Tension-network connectivity. Models spider web
    geometry: radial threads (long-range) plus spiral
    threads (local). Connectivity emerges from tension
    balance. Perturbation at any node propagates
    through entire structure.

    Spin: DOWN — tension matrix damps oscillations
    Phi note: C5 (6.854)

    Returns scalar connectivity value in [0,1].
    """
    # Web connectivity scales with thread count
    total_threads = n_radial + n_spiral
    max_threads   = (n_radial * n_spiral)

    # Phi-scaled tension ratio
    tension_ratio = phi_scale(
        n_radial / total_threads, 4
    ) / PHI_SERIES[4]

    # Web matures over time then degrades with perturbation
    maturation  = 1 - np.exp(-0.04 * t)
    perturbation = np.exp(-0.001 * t**1.2)

    G = tension_ratio * maturation * perturbation
    return float(np.clip(G, 0, 1))


def G_bee(t, n_scouts=20, threshold=0.6):
    """
    Bee — Swarm Dynamics
    Emergent global structure from local rules.
    Models waggle dance information propagation:
    scouts find resources, encode distance/direction,
    recruit others. No central coordination.
    Resilient to node loss, sensitive to rule disruption.

    Spin: UP — swarm amplifies successful signals
    Phi note: G5 (base of lower series)

    Returns scalar connectivity value in [0,1].
    """
    np.random.seed(int(t * 13) % 10000)

    # Scout discovery probability increases with time
    # then saturates as environment is mapped
    discovery_rate = sigmoid(
        phi_scale(0.03, 1) * t - 2
    )

    # Recruitment cascade
    active_scouts = np.random.binomial(
        n_scouts,
        discovery_rate
    )
    recruited     = active_scouts * PHI

    # Quorum threshold: below threshold, no coordinated action
    if active_scouts / n_scouts < (1 - threshold):
        G = active_scouts / n_scouts * 0.3
    else:
        G = min(recruited / n_scouts, 1.0)

    return float(np.clip(G, 0, 1))


def G_elephant(t, memory_depth=50):
    """
    Elephant — Memory Algorithms
    History-weighted persistent edges. Models elephant
    spatial memory: paths used historically remain
    stronger than unused paths. Multigenerational
    knowledge encoded in connectivity weights.
    Edges never fully disappear — only fade.

    Spin: DOWN then UP — long consolidation then
    persistent high connectivity
    Phi note: F5 (11.090)

    Returns scalar connectivity value in [0,1].
    """
    # Memory kernel: exponential with very long tail
    # Elephant memory integrates over phi_scale(memory_depth,5) years
    tau = phi_scale(memory_depth, 5)

    # Connectivity grows logarithmically — slow but persistent
    G_base = np.log(1 + t / tau) / np.log(
        1 + 100 / tau
    )

    # Long-term stability: once established, very resistant to loss
    stability = 1 - np.exp(
        -phi_scale(0.01, 5) * t
    )

    G = G_base * (ALPHA_OLD + ALPHA_NEW * stability)
    return float(np.clip(G, 0, 1))


def G_wolf(t, pack_size=8, cohesion=0.7):
    """
    Wolf — Pack Coordination
    Dynamic role-based network. Models wolf pack
    structure: alpha/beta/omega hierarchy creates
    directed connectivity. Coordinated hunting
    requires synchronization across roles.
    Connectivity peaks during coordinated action.

    Spin: UP — coordinated bursts increase effective
    connectivity transiently
    Phi note: A5 (pack coordination)

    Returns scalar connectivity value in [0,1].
    """
    np.random.seed(int(t * 17) % 10000)

    # Pack cohesion oscillates with hunt cycles
    hunt_cycle = T_C / PHI_SERIES[2]  # ~22.9 year cycle
    phase_sync = (
        np.cos(2 * np.pi * t / hunt_cycle) + 1
    ) / 2

    # Cohesion builds with pack experience
    experience = 1 - np.exp(-0.03 * t)

    # Effective connectivity: roles × cohesion × phase
    role_factor = np.log(pack_size) / np.log(
        pack_size + 1
    )
    G = role_factor * cohesion * experience * (
        0.5 + 0.5 * phase_sync
    )

    return float(np.clip(G, 0, 1))


# ── Mixture model ──────────────────────────────────────────

# Default weights — sum to 1
# Ordered: Raven, Octopus, Dolphin, Parrot,
#          Spider, Bee, Elephant, Wolf
DEFAULT_WEIGHTS = np.array([
    1/PHI**2,   # Raven    — foundational
    1/PHI**3,   # Octopus  — combinatorial
    1/PHI**1,   # Dolphin  — harmonic (strongest)
    1/PHI**3,   # Parrot   — adaptive
    1/PHI**4,   # Spider   — damping
    1/PHI**2,   # Bee      — emergent
    1/PHI**1,   # Elephant — persistent (strongest)
    1/PHI**2,   # Wolf     — coordinated
])
DEFAULT_WEIGHTS /= DEFAULT_WEIGHTS.sum()


def compute_C_new(t, state=None, weights=None,
                  gamma=0.3):
    """
    Main entry point for integrative_sim_v2.py

    Computes emergent animal-lattice connectivity
    at time t as weighted mixture of eight geometries.

    Parameters
    ----------
    t       : float, current time step
    state   : dict, optional simulation state
              (used by Parrot for history)
    weights : array-like, optional mixture weights
              defaults to phi-scaled DEFAULT_WEIGHTS
    gamma   : float, scaling factor for C_new
              relative to ac(t)

    Returns
    -------
    C_new   : float in [0, 1]
    G_vals  : dict of individual geometry values
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    else:
        weights = np.array(weights)
        weights /= weights.sum()

    # Extract parrot history from state if available
    parrot_history = None
    if state is not None and 'C_new_history' in state:
        parrot_history = state['C_new_history']

    # Evaluate all eight geometries
    G_vals = {
        'raven':    G_raven(t),
        'octopus':  G_octopus(t),
        'dolphin':  G_dolphin(t),
        'parrot':   G_parrot(t, parrot_history),
        'spider':   G_spider(t),
        'bee':      G_bee(t),
        'elephant': G_elephant(t),
        'wolf':     G_wolf(t)
    }

    G_array = np.array(list(G_vals.values()))

    # Weighted mixture
    G_mix = np.dot(weights, G_array)

    # Growth envelope: lattice spins up from C_seed
    growth = C_SEED * (
        1 - np.exp(-0.05 * t)
    ) * G_mix

    C_new = float(np.clip(growth, 0, 1))

    return C_new, G_vals


# ── Survival diagnostic ────────────────────────────────────

def survival_window(time_series, C_total_series,
                    met_mult_series, baseline=1.0):
    """
    Track time intervals where survival condition holds:

        C_total(t) > baseline * met_mult(t)

    Parameters
    ----------
    time_series     : array of time values
    C_total_series  : array of total connectivity values
    met_mult_series : array of metabolic multipliers
    baseline        : float, reference connectivity

    Returns
    -------
    dict with:
        condition   : boolean array
        window_length : total years condition holds
        first_failure : time of first failure
        noah_ark    : boolean, does fast cycler
                      find viable window after slow fails
    """
    condition = np.array(C_total_series) > (
        baseline * np.array(met_mult_series)
    )

    window_length = np.sum(condition) * (
        time_series[1] - time_series[0]
    )

    failures = np.where(~condition)[0]
    first_failure = (
        time_series[failures[0]]
        if len(failures) > 0 else None
    )

    return {
        'condition':      condition,
        'window_length':  window_length,
        'first_failure':  first_failure,
        'pct_viable':     100 * np.mean(condition)
    }


# ── Visualization ──────────────────────────────────────────

def plot_symphony(T=300, dt=0.5):
    """
    Plot all eight geometries over time.
    Shows individual contributions and mixture.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    time = np.arange(0, T, dt)
    n    = len(time)

    # Evaluate all geometries
    geometry_series = {k: np.zeros(n) for k in [
        'raven','octopus','dolphin','parrot',
        'spider','bee','elephant','wolf'
    ]}
    C_new_series = np.zeros(n)
    state = {'C_new_history': []}

    for i, t in enumerate(time):
        C_new, G_vals = compute_C_new(t, state)
        for k, v in G_vals.items():
            geometry_series[k][i] = v
        C_new_series[i] = C_new
        state['C_new_history'].append(C_new)

    # Old percolation connectivity
    k_perc = 0.4
    ac_series = 1 / (
        1 + np.exp(k_perc * (time - T_C))
    )

    # Total connectivity
    gamma = 0.3
    C_total = ac_series + gamma * C_new_series

    # Colors mapped to Lattice Symphony image
    colors = {
        'raven':    '#00ffcc',
        'octopus':  '#00cc88',
        'dolphin':  '#4488ff',
        'parrot':   '#ff4422',
        'spider':   '#ffaa00',
        'bee':      '#ffdd00',
        'elephant': '#aaaaaa',
        'wolf':     '#ff88aa'
    }

    notes = {
        'raven':    '1Φ',
        'octopus':  'A4 2.618Φ',
        'dolphin':  '1.618Φ',
        'parrot':   'D5 4.236Φ',
        'spider':   'C5 6.854Φ',
        'bee':      'G5',
        'elephant': 'F5 11.090Φ',
        'wolf':     'A5'
    }

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#080818')

    gs = gridspec.GridSpec(
        4, 4, figure=fig,
        hspace=0.5, wspace=0.4
    )

    # Individual geometry panels
    positions = [
        (0,0), (0,1), (0,2), (0,3),
        (1,0), (1,1), (1,2), (1,3)
    ]
    animals = list(geometry_series.keys())

    for idx, (animal, pos) in enumerate(
        zip(animals, positions)
    ):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        ax.set_facecolor('#0a0a1a')
        ax.plot(
            time,
            geometry_series[animal],
            color=colors[animal],
            linewidth=1.5
        )
        ax.set_title(
            f'{animal.capitalize()}\n{notes[animal]}',
            color=colors[animal],
            fontsize=8
        )
        ax.set_ylim(0, 1)
        ax.tick_params(colors='#666666', labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor('#222233')
        ax.grid(True, color='#222233',
                linewidth=0.5, alpha=0.5)

    # Main panel: all geometries + mixture + total
    ax_main = fig.add_subplot(gs[2, :])
    ax_main.set_facecolor('#0a0a1a')

    for animal in animals:
        ax_main.plot(
            time,
            geometry_series[animal],
            color=colors[animal],
            linewidth=1, alpha=0.5,
            label=f'{animal} {notes[animal]}'
        )

    ax_main.plot(
        time, C_new_series,
        color='white', linewidth=2.5,
        label='G_mix (weighted mixture)',
        zorder=10
    )
    ax_main.plot(
        time, ac_series,
        color='#ff4444', linewidth=2,
        linestyle='--',
        label='ac(t) old habitat network'
    )
    ax_main.plot(
        time, C_total,
        color='#44ff88', linewidth=2.5,
        label='C_total = ac + γ·C_new',
        zorder=11
    )

    ax_main.axvline(
        x=T_C, color='red',
        linestyle=':', alpha=0.7
    )
    ax_main.annotate(
        f't_c = {T_C}',
        (T_C, 0.85),
        color='red', fontsize=8
    )
    ax_main.set_xlabel('Time', color='#aaaaaa')
    ax_main.set_ylabel('Connectivity', color='#aaaaaa')
    ax_main.set_title(
        'Lattice Symphony — Full mixture vs decaying habitat',
        color='white', fontsize=10
    )
    ax_main.legend(
        loc='upper right', fontsize=6,
        facecolor='#111122', labelcolor='white',
        ncol=2
    )
    ax_main.tick_params(colors='#666666')
    for spine in ax_main.spines.values():
        spine.set_edgecolor('#222233')
    ax_main.grid(
        True, color='#222233',
        linewidth=0.5, alpha=0.5
    )

    # Survival window panel
    ax_surv = fig.add_subplot(gs[3, :])
    ax_surv.set_facecolor('#0a0a1a')

    from core import metabolic_multiplier
    import json
    with open('../Model/parameters.json') as f:
        P = json.load(f)

    met_mults = np.array([
        metabolic_multiplier(
            P['Q10_apex'],
            P['WARM_A'] * t + P['WARM_B'] * t**2
            if 'WARM_A' in P else 0.02 * t
        )
        for t in time
    ])

    # Survival conditions for slow and fast
    baseline_slow = 0.6
    baseline_fast = 0.3

    surv_slow = survival_window(
        time, C_total, met_mults, baseline_slow
    )
    surv_fast = survival_window(
        time, C_total, met_mults, baseline_fast
    )

    ax_surv.fill_between(
        time, 0, 1,
        where=surv_slow['condition'],
        alpha=0.3, color='darkblue',
        label=f"Slow viable: {surv_slow['pct_viable']:.1f}%"
    )
    ax_surv.fill_between(
        time, 0, 1,
        where=surv_fast['condition'],
        alpha=0.3, color='darkorange',
        label=f"Fast viable: {surv_fast['pct_viable']:.1f}%"
    )
    ax_surv.plot(
        time, C_total,
        color='#44ff88', linewidth=1.5
    )
    ax_surv.plot(
        time, met_mults * baseline_slow,
        color='darkblue', linewidth=1.5,
        linestyle='--', label='Slow threshold'
    )
    ax_surv.plot(
        time, met_mults * baseline_fast,
        color='darkorange', linewidth=1.5,
        linestyle='--', label='Fast threshold'
    )

    if surv_slow['first_failure']:
        ax_surv.axvline(
            x=surv_slow['first_failure'],
            color='darkblue',
            linestyle=':', alpha=0.8
        )
        ax_surv.annotate(
            f"Slow fails\nt={surv_slow['first_failure']:.0f}",
            (surv_slow['first_failure'], 0.7),
            color='lightblue', fontsize=7
        )

    ax_surv.set_xlabel('Time', color='#aaaaaa')
    ax_surv.set_ylabel(
        'Connectivity / Threshold',
        color='#aaaaaa'
    )
    ax_surv.set_title(
        'Survival window diagnostic — '
        'Noah\'s Ark condition',
        color='white', fontsize=10
    )
    ax_surv.legend(
        loc='upper right', fontsize=7,
        facecolor='#111122', labelcolor='white'
    )
    ax_surv.tick_params(colors='#666666')
    for spine in ax_surv.spines.values():
        spine.set_edgecolor('#222233')
    ax_surv.grid(
        True, color='#222233',
        linewidth=0.5, alpha=0.5
    )

    fig.suptitle(
        'Lattice Symphony — Animal Geometry Basis Functions\n'
        'Evolutionary rescue via structured connectivity',
        color='white', fontsize=12, y=0.99
    )

    plt.savefig(
        'lattice_symphony_output.png',
        dpi=150,
        facecolor=fig.get_facecolor()
    )
    plt.show()

    # Print summary
    print("\n── Lattice Symphony Summary ──")
    print(f"{'Animal':<12} {'Mean G':>8} "
          f"{'Max G':>8} {'Phi note':<12}")
    print("─" * 44)
    for animal in animals:
        series = geometry_series[animal]
        print(
            f"{animal:<12} "
            f"{np.mean(series):>8.3f} "
            f"{np.max(series):>8.3f} "
            f"{notes[animal]:<12}"
        )
    print(f"\n{'C_new mean':<20} {np.mean(C_new_series):.3f}")
    print(f"{'C_total mean':<20} {np.mean(C_total):.3f}")
    print(
        f"{'Slow viable':<20} "
        f"{surv_slow['pct_viable']:.1f}%"
    )
    print(
        f"{'Fast viable':<20} "
        f"{surv_fast['pct_viable']:.1f}%"
    )


if __name__ == '__main__':
    plot_symphony()
