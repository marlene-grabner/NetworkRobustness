"""
Reference tests: every metric and algorithm that is implemented in this
package is compared to an independent result -- an established library
(sklearn, networkx), a direct dense calculation, or a case small enough to
work out by hand. The project's own conventions (degenerate partitions, nodes
without edges, ties) are pinned down with hand-made cases.
"""

import networkx as nx
import numpy as np
import pytest
from scipy import sparse
from scipy.stats import hypergeom
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    roc_auc_score,
)

from NoiseEffect.CommunityDetection import ALGORITHMS, detect_partitions
from NoiseEffect.CommunityDetection import similarity as sim
from NoiseEffect.GlobalProperties import metrics as m
from NoiseEffect.GlobalProperties.fiedler_value import fiedler_on_gcc
from NoiseEffect.SeedExpansion import SyntheticSeeds as synth
from NoiseEffect.SeedExpansion.Expansion import comparing as cmp
from NoiseEffect.SeedExpansion.Expansion import edges_to_adjacency, run_algorithm
from NoiseEffect.SeedExpansion.Expansion.Algorithms import (
    diamond_score,
    first_neighbors_score,
    rwr_row_normalized,
    rwr_symmetric_normalized,
)
from NoiseEffect.TopologicalProperties import get_network_profile

NAN = float("nan")


def to_edges(g: nx.Graph) -> np.ndarray:
    return np.array(g.edges(), dtype=np.int64).reshape(-1, 2)


def to_adj(g: nx.Graph) -> sparse.csr_matrix:
    return edges_to_adjacency(to_edges(g), g.number_of_nodes())


@pytest.fixture(scope="module")
def fragmented() -> nx.Graph:
    """200 nodes: a large component, several small ones and nodes without edges."""
    g = nx.gnp_random_graph(200, 0.012, seed=4)
    sizes = sorted(len(c) for c in nx.connected_components(g))
    assert sizes[0] == 1 and sizes[-1] > 50 and len(sizes) > 5
    return g


@pytest.fixture(scope="module")
def scale_free() -> nx.Graph:
    return nx.barabasi_albert_graph(300, 3, seed=2)


############################################################
# Community comparison


def test_ari_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(2, 200)
        a = rng.integers(0, rng.integers(1, 12), n)
        b = rng.integers(0, rng.integers(1, 12), n)
        assert sim.adjusted_rand_index(a, b) == pytest.approx(adjusted_rand_score(a, b), abs=1e-12)


@pytest.mark.parametrize(
    "a, b, b_is_baseline, expected",
    [
        # two real partitions that agree
        ([0, 0, 1, 1], [5, 5, 3, 3], True, 1.0),
        # the perturbed network collapsed (one community / all alone): ARI 0
        ([0, 0, 0, 0], [0, 0, 1, 1], True, 0.0),
        ([0, 1, 2, 3], [0, 0, 1, 1], True, 0.0),
        # the baseline itself is degenerate: the pair is excluded
        ([0, 0, 1, 1], [0, 0, 0, 0], True, NAN),
        ([0, 0, 0, 0], [0, 0, 0, 0], True, NAN),
        # between seeds: 0 if one is degenerate, excluded only if both are
        ([0, 0, 1, 1], [0, 0, 0, 0], False, 0.0),
        ([0, 0, 0, 0], [0, 0, 1, 1], False, 0.0),
        ([0, 0, 0, 0], [0, 1, 2, 3], False, NAN),
    ],
)
def test_ari_of_degenerate_partitions(a, b, b_is_baseline, expected):
    all_nodes, connected = sim.ari_all_and_connected(np.array(a), np.array(b), b_is_baseline)
    assert all_nodes == pytest.approx(expected, nan_ok=True)
    assert connected == pytest.approx(expected, nan_ok=True)


def test_ari_with_nodes_without_edges():
    baseline = np.array([0, 0, 0, 1, 1, 1])
    perturbed = np.array([0, 0, -1, 1, 1, -1])  # two nodes lost all their edges
    all_nodes, connected = sim.ari_all_and_connected(perturbed, baseline, True)
    # connected: the four remaining nodes are still split the same way
    assert connected == 1.0
    # all: each lost node is a community of its own, so structure was lost
    assert all_nodes == pytest.approx(adjusted_rand_score([0, 0, 2, 1, 1, 3], baseline))
    assert all_nodes < 1.0


def test_mean_ari():
    good = np.array([[0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0]])
    collapsed = np.array([[0, 0, 0, 0, 0, 0]])
    assert sim.mean_ari(good) == (1.0, 1.0, 1)
    assert sim.mean_ari(good, good) == (1.0, 1.0, 4)
    # one seed of the perturbed network collapsed: it counts as 0, not as missing
    mixed = np.vstack([good[:1], collapsed])
    assert sim.mean_ari(mixed, good) == (0.5, 0.5, 4)
    # a degenerate baseline leaves no valid pair
    all_nodes, connected, n_valid = sim.mean_ari(good, collapsed)
    assert np.isnan(all_nodes) and np.isnan(connected) and n_valid == 0


def test_collapse_statistics():
    labels = np.array(
        [
            [0] * 20 + [-1],  # one community
            [0] * 19 + [1] + [-1],  # 95% in the largest community
            [0] * 10 + [1] * 10 + [-1],  # two equal communities
            list(range(20)) + [-1],  # every node alone
        ]
    )
    assert sim.collapse_statistics(labels) == {
        "n_connected_nodes": 20,
        "n_communities": (1 + 2 + 2 + 20) / 4,
        "largest_community_frac": pytest.approx((1 + 0.95 + 0.5 + 0.05) / 4),
        "n_collapsed": 1,
        "n_near_collapsed": 1,
        "n_all_singletons": 1,
    }


############################################################
# Community detection


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
def test_planted_communities_are_found(algorithm, toy):
    labels = detect_partitions(toy["edges"], toy["n_nodes"], algorithm, seeds=(1, 2, 3))
    assert labels.shape == (3, toy["n_nodes"])
    isolated = np.array([toy["node_to_idx"][n] for n in ("z_isolated_1", "z_isolated_2")])
    assert (labels[:, isolated] == -1).all()
    connected = np.delete(labels, isolated, axis=1)
    assert (connected >= 0).all()
    for seed_labels in connected:
        assert adjusted_rand_score(toy["truth"], seed_labels) > 0.9


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
def test_community_detection_is_reproducible(algorithm, toy):
    first = detect_partitions(toy["edges"], toy["n_nodes"], algorithm, seeds=(1, 2))
    second = detect_partitions(toy["edges"], toy["n_nodes"], algorithm, seeds=(1, 2))
    assert (first == second).all()


############################################################
# Global properties


def test_gcc_and_singletons(fragmented):
    g = m.build_graph(to_edges(fragmented), 200)
    largest = max(len(c) for c in nx.connected_components(fragmented))
    assert m.gcc_and_singletons(g) == {
        "gcc": largest / 200,
        "num_singletons": sum(d == 0 for _, d in fragmented.degree()),
    }


def test_build_graph_ignores_duplicate_edges_and_self_loops():
    g = m.build_graph(np.array([[0, 1], [1, 0], [0, 1], [2, 2]]), 4)
    assert g.vcount() == 4 and g.get_edgelist() == [(0, 1)]


@pytest.mark.parametrize("graph", ["fragmented", "scale_free"])
def test_exact_global_efficiency_matches_networkx(graph, request):
    g = request.getfixturevalue(graph)
    ig_graph = m.build_graph(to_edges(g), g.number_of_nodes())
    exact = m.global_efficiency(ig_graph, n_sources=None)["efficiency"]
    assert exact == pytest.approx(nx.global_efficiency(g), rel=1e-12)


def test_sampled_global_efficiency_is_close_to_exact(fragmented, scale_free):
    for g in (fragmented, scale_free):
        ig_graph = m.build_graph(to_edges(g), g.number_of_nodes())
        exact = m.global_efficiency(ig_graph, n_sources=None)["efficiency"]
        sampled = m.global_efficiency(ig_graph, n_sources=40)["efficiency"]
        assert sampled != exact  # the sampling is really used
        assert sampled == pytest.approx(exact, rel=0.05)


def test_simple_metrics_match_networkx(scale_free):
    g = m.build_graph(to_edges(scale_free), 300)
    assert m.transitivity(g)["transitivity"] == pytest.approx(nx.transitivity(scale_free))
    assert m.assortativity(g)["assortativity"] == pytest.approx(
        nx.degree_assortativity_coefficient(scale_free)
    )
    core = np.array(list(nx.core_number(scale_free).values()))
    assert m.kcore(g) == {
        "max_core": core.max(),
        "max_core_frac": pytest.approx(np.mean(core == core.max())),
    }
    dense = nx.to_numpy_array(scale_free)
    assert m.largest_eigenvalue(g)["largest_eigenvalue"] == pytest.approx(
        np.linalg.eigvalsh(dense)[-1]
    )


def test_centrality_rank(scale_free):
    g = m.build_graph(to_edges(scale_free), 300)
    reference = m.centralities(g)
    assert (reference["degree"] == [d for _, d in scale_free.degree()]).all()
    pagerank = np.array(list(nx.pagerank(scale_free, tol=1e-12).values())) * 300
    assert reference["pagerank"] == pytest.approx(pagerank, abs=1e-5)

    same = m.centrality_rank(g, reference)
    assert set(same.values()) == {1.0}

    # removing all edges of the 3 largest hubs must be visible in the top 1% (3 nodes)
    hubs = np.argsort(-reference["degree"], kind="stable")[:3]
    edges = to_edges(scale_free)
    kept = edges[~np.isin(edges, hubs).any(axis=1)]
    changed = m.centrality_rank(m.build_graph(kept, 300), reference)
    assert changed["degree_top1_overlap"] == 0.0
    assert changed["degree_rank_corr"] < 1.0
    assert 0.0 < changed["degree_top10_overlap"] < 1.0


def dense_fiedler(g: nx.Graph) -> float:
    gcc = g.subgraph(max(nx.connected_components(g), key=len))
    return float(np.linalg.eigvalsh(nx.laplacian_matrix(gcc).toarray().astype(float))[1])


@pytest.mark.parametrize(
    "g, disconnected",
    [
        (nx.gnp_random_graph(300, 0.05, seed=1), False),  # dense: fast solver
        (nx.grid_2d_graph(30, 30), False),  # sparse, grid-like: exact fallback
        (nx.gnp_random_graph(200, 0.012, seed=4), True),  # largest component only
        (nx.path_graph(10), False),  # small: dense solver
        (nx.disjoint_union(nx.complete_graph(6), nx.path_graph(3)), True),
    ],
    ids=["dense", "grid", "fragmented", "small", "small_disconnected"],
)
def test_fiedler_value_matches_dense_eigenvalues(g, disconnected):
    g = nx.convert_node_labels_to_integers(g)
    A = nx.to_scipy_sparse_array(g, format="csr").astype(float)
    value, was_disconnected = fiedler_on_gcc(A, A.shape[0])
    assert was_disconnected is disconnected
    assert value == pytest.approx(dense_fiedler(g), rel=1e-4)


def test_fiedler_value_of_long_chains_uses_the_exact_solver():
    """
    On long, chain-like networks (power grid, roads) the fast solver stops
    early at a value that is several times too large; its result must be
    rejected. The exact values are known: 2 (1 - cos(pi / n)) for a path and
    2 (1 - cos(2 pi / n)) for a cycle.
    """
    for g, expected in (
        (nx.path_graph(2000), 2 * (1 - np.cos(np.pi / 2000))),
        (nx.cycle_graph(3000), 2 * (1 - np.cos(2 * np.pi / 3000))),
    ):
        A = nx.to_scipy_sparse_array(g, format="csr").astype(float)
        value, _ = fiedler_on_gcc(A, A.shape[0])
        assert value == pytest.approx(expected, rel=1e-4)


def test_network_profile(fragmented):
    profile = get_network_profile(fragmented)
    gcc = fragmented.subgraph(max(nx.connected_components(fragmented), key=len))
    assert profile["Nodes_Full"] == 200
    assert profile["Edges_Full"] == fragmented.number_of_edges()
    assert profile["N_Components"] == nx.number_connected_components(fragmented)
    assert profile["N_Singletons"] == sum(d == 0 for _, d in fragmented.degree())
    assert profile["GCC_Fraction"] == len(gcc) / 200
    # the structural metrics are those of the largest component
    assert profile["Metrics_Scope"] == "GCC"
    assert profile["Nodes"] == len(gcc)
    assert profile["Average_Degree"] == pytest.approx(2 * gcc.number_of_edges() / len(gcc))
    assert profile["Transitivity"] == pytest.approx(nx.transitivity(gcc))
    assert profile["Avg_Path_Length"] == pytest.approx(nx.average_shortest_path_length(gcc))
    assert profile["Algebraic_Connectivity"] == pytest.approx(dense_fiedler(fragmented), rel=1e-4)


############################################################
# Seed expansion algorithms

SEEDS = np.array([0, 5, 17])
RESTART = 0.7


def test_rwr_matches_the_closed_form(scale_free):
    A = nx.to_numpy_array(scale_free)
    d = A.sum(axis=1)
    p0 = np.zeros(300)
    p0[SEEDS] = 1 / len(SEEDS)
    not_seed = np.ones(300, dtype=bool)
    not_seed[SEEDS] = False
    identity = np.eye(300)

    # p = restart * (I - (1 - restart) W)^-1 p0
    row = RESTART * np.linalg.solve(identity - (1 - RESTART) * A / d, p0)
    sym = RESTART * np.linalg.solve(identity - (1 - RESTART) * A / np.sqrt(np.outer(d, d)), p0)
    assert row.sum() == pytest.approx(1.0)  # a probability distribution

    adj = to_adj(scale_free)
    for function, expected in ((rwr_row_normalized, row), (rwr_symmetric_normalized, sym)):
        scores = function(adj, SEEDS, restart=RESTART)
        assert (scores[SEEDS] == -np.inf).all()
        assert scores[not_seed] == pytest.approx(expected[not_seed], abs=1e-7)


def test_rwr_does_not_reach_other_components(fragmented):
    adj = to_adj(fragmented)
    component = max(nx.connected_components(fragmented), key=len)
    seeds = np.array(sorted(component)[:3])
    outside = np.array(sorted(set(fragmented) - component))
    for name in ("rwr_row", "rwr_sym"):
        scores = run_algorithm(name, adj, seeds)
        assert (scores[outside] == 0).all()
        assert (scores[np.array(sorted(component - set(seeds.tolist())))] > 0).all()


def test_first_neighbors_counts_edges_to_the_seeds(scale_free):
    adj = to_adj(scale_free)
    scores = first_neighbors_score(adj, SEEDS)
    seeds = set(SEEDS.tolist())
    for node in scale_free:
        if node in seeds:
            assert scores[node] == -np.inf
            continue
        links = len(seeds & set(scale_free[node]))
        assert round(scores[node]) == links
        # only seed neighbours count as found (score > 0)
        assert (scores[node] > 0) == (links > 0)

    # equal number of links: the node with the lower degree comes first
    star = nx.Graph([(0, 1), (0, 2), (2, 3), (2, 4)])
    scores = first_neighbors_score(to_adj(star), np.array([0]))
    assert scores[1] > scores[2] > 0


def reference_diamond(g: nx.Graph, seeds, n_add: int) -> list:
    """DIAMOnD as in Ghiassian et al. 2015, without any shortcut: every
    candidate's p-value in every iteration; ties go to the lowest node index."""
    module = set(seeds)
    n, added = g.number_of_nodes(), []
    for _ in range(n_add):
        best = None
        for node in sorted(set(g) - module):
            links = len(module & set(g[node]))
            if links == 0:
                continue
            p = hypergeom.sf(links - 1, n, len(module), g.degree(node))
            if best is None or p < best[0]:
                best = (p, node)
        if best is None:
            break
        added.append(best[1])
        module.add(best[1])
    return added


@pytest.mark.parametrize("graph", ["scale_free", "fragmented", "toy_graph"])
def test_diamond_matches_the_reference_implementation(graph, request):
    g = nx.convert_node_labels_to_integers(request.getfixturevalue(graph), ordering="sorted")
    seeds = np.array(sorted(max(nx.connected_components(g), key=len))[:4])
    n_add = 40
    scores = diamond_score(to_adj(g), seeds, max_added_nodes=n_add)

    expected = reference_diamond(g, seeds.tolist(), n_add)
    assert len(expected) > 10
    added = np.flatnonzero(scores > 0)
    order = added[np.argsort(-scores[added])].tolist()
    assert order == expected
    # first added = highest score; seeds -inf; everything else 0
    assert scores[expected[0]] == n_add
    assert (scores[seeds] == -np.inf).all()
    assert np.sum(scores == 0) == g.number_of_nodes() - len(seeds) - len(expected)


############################################################
# Seed expansion metrics


def test_auroc_and_auprc_match_sklearn_with_ties_and_seeds():
    rng = np.random.default_rng(1)
    for _ in range(100):
        n = 120
        scores = np.round(rng.random(n), 1)  # many ties
        scores[rng.choice(n, 30, replace=False)] = 0.0  # nodes that were not reached
        seeds = rng.choice(n, 10, replace=False)
        scores[seeds] = -np.inf
        candidates = np.setdiff1d(np.arange(n), seeds)
        positives = rng.choice(candidates, 15, replace=False)

        truth = np.isin(candidates, positives)
        auroc, auprc = cmp.RankedScores(scores).auroc_auprc(positives)
        assert auroc == pytest.approx(roc_auc_score(truth, scores[candidates]), abs=1e-12)
        assert auprc == pytest.approx(
            average_precision_score(truth, scores[candidates]), abs=1e-12
        )


def test_auroc_ignores_positives_that_are_seeds_and_needs_both_classes():
    scores = np.array([-np.inf, 3.0, 2.0, 1.0, 0.0])
    ranked = cmp.RankedScores(scores)
    assert ranked.auroc_auprc(np.array([0, 1])) == (1.0, 1.0)  # node 0 is a seed
    assert ranked.auroc_auprc(np.array([4])) == (0.0, 0.25)
    assert np.isnan(ranked.auroc_auprc(np.array([0]))).all()  # no positive left
    assert np.isnan(ranked.auroc_auprc(np.array([1, 2, 3, 4]))).all()  # no negative


def test_top_valid_matches_a_full_sort():
    rng = np.random.default_rng(2)
    for _ in range(100):
        scores = np.round(rng.random(60), 1)
        scores[rng.choice(60, 20, replace=False)] = 0.0
        scores[rng.choice(60, 5, replace=False)] = -np.inf
        valid = [i for i in range(60) if scores[i] > 0]
        expected = sorted(valid, key=lambda i: (-scores[i], i))
        for k in (1, 5, 20, 100):
            assert cmp.top_valid(scores, k).tolist() == expected[:k]


def test_ties_at_the_cut_off_are_counted():
    scores = np.array([5.0, 4.0, 4.0, 4.0, 3.0, 0.0, -np.inf])
    top = cmp.top_valid(scores, 5)
    assert top.tolist() == [0, 1, 2, 3, 4]  # nodes with score 0 are never in the top
    assert cmp.n_tied_outside(scores, top, 2) == 2  # nodes 2 and 3 lost the tie-break
    assert cmp.n_tied_outside(scores, top, 4) == 0
    assert cmp.n_tied_outside(scores, top, 10) == 0  # fewer than k valid nodes


def test_overlap_metrics():
    assert cmp.overlap_metrics({1, 2, 3, 4}, {3, 4, 5}) == {
        "jaccard": 2 / 5,
        "precision": 2 / 3,
        "recall": 2 / 4,
        "f1": pytest.approx(2 * (2 / 3) * (1 / 2) / (2 / 3 + 1 / 2)),
    }
    # seeds cut off from the network: nothing is found, which is a 0 and not missing
    assert cmp.overlap_metrics({1, 2}, set()) == {
        "jaccard": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0
    }


def test_recovery_metrics():
    #                 seed     hidden: 1, 3, 6            not reached
    scores = np.array([-np.inf, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.0])
    hidden = np.array([1, 3, 6])
    ranked = cmp.RankedScores(scores)
    top = cmp.top_valid(scores, 4)
    rows = cmp.recovery_metrics(scores, ranked, top, hidden, k_list=[2, 4])
    assert [(r["k"], r["recovery_recall"], r["recovery_precision"]) for r in rows] == [
        (2, 1 / 3, 1 / 2),
        (4, 2 / 3, 2 / 4),
    ]
    # R-precision: hidden nodes in the top 3 (= number of hidden nodes)
    assert {r["recovery_r_precision"] for r in rows} == {2 / 3}
    truth = np.isin(np.arange(1, 8), hidden)
    assert rows[0]["recovery_auroc"] == pytest.approx(roc_auc_score(truth, scores[1:]))


############################################################
# Synthetic seed sets


def test_bfs_modules_are_connected_and_complete(toy_graph, fragmented):
    g = nx.convert_node_labels_to_integers(toy_graph, ordering="sorted")
    adj, rng = to_adj(g), np.random.default_rng(0)
    for start in (0, 40, 119):
        module = synth.bfs_module(adj, start, 25, rng)
        assert module[0] == start and len(set(module.tolist())) == 25
        assert nx.is_connected(g.subgraph(module.tolist()))
        # BFS takes all nodes of a distance before the next one
        distance = nx.single_source_shortest_path_length(g, start)
        cutoff = max(distance[n] for n in module.tolist())
        closer = {n for n, d in distance.items() if d < cutoff}
        assert closer <= set(module.tolist())

    # a component smaller than the module is rejected
    small = min(nx.connected_components(fragmented), key=len)
    assert synth.bfs_module(to_adj(fragmented), next(iter(small)), 10, rng) is None


def test_split_module():
    rng = np.random.default_rng(0)
    module = np.arange(100, 120)
    seeds, hidden = synth.split_module(module, 0.25, rng)
    assert len(seeds) == 5 and len(hidden) == 15
    assert sorted(np.concatenate([seeds, hidden]).tolist()) == module.tolist()
    # both parts always have at least one node
    assert [len(part) for part in synth.split_module(module, 0.0, rng)] == [1, 19]
    assert [len(part) for part in synth.split_module(module, 1.0, rng)] == [19, 1]


def test_bfs_seed_sets_respect_the_overlap_limit_and_target_degree(scale_free):
    adj = to_adj(scale_free)
    records = synth.bfs_seed_sets(
        adj, [0.1, 0.9], n_sets=4, module_size=15, seed_fraction=0.3,
        max_overlap=0.2, rng=np.random.default_rng(0),
    )
    assert [r["target_percentile"] for r in records] == [0.1] * 4 + [0.9] * 4
    modules = [np.concatenate([r["seeds"], r["hidden"]]) for r in records]
    for i, module in enumerate(modules):
        assert len(module) == 15 and len(records[i]["seeds"]) == 4
        assert records[i]["start_node"] in module
        for other in modules[:i]:
            assert synth.jaccard(module, other) <= 0.2
    low = np.mean([r["start_degree"] for r in records[:4]])
    high = np.mean([r["start_degree"] for r in records[4:]])
    assert low < high


def test_large_communities_are_split_inside_their_community(toy):
    adj = edges_to_adjacency(toy["edges"], toy["n_nodes"])
    labels = np.full(toy["n_nodes"], -1)
    labels[:60], labels[60:120] = 0, 1  # two communities of two planted groups each
    new_labels, parents, depths = synth.split_large_communities(adj, labels, max_size=40, seed=1)

    assert ((new_labels == -1) == (labels == -1)).all()
    sizes = np.bincount(new_labels[new_labels >= 0])
    assert len(sizes) == len(parents) == len(depths) >= 4
    assert sizes.max() <= 40 and (depths >= 1).all()
    for piece, parent in enumerate(parents):
        assert (labels[new_labels == piece] == parent).all()

    # nothing to split: the labels stay as they are
    unchanged, parents, depths = synth.split_large_communities(adj, labels, max_size=60, seed=1)
    assert (unchanged == labels).all() and parents.tolist() == [0, 1] and depths.tolist() == [0, 0]


def test_community_seed_sets_use_only_communities_of_the_right_size():
    labels = np.repeat([0, 1, 2, 3, 4, -1], [5, 20, 25, 30, 80, 10])
    records = synth.community_seed_sets(
        labels, n_sets=2, min_size=10, max_size=40, seed_fraction=0.2,
        rng=np.random.default_rng(0),
    )
    assert len(records) == 2
    for record in records:
        assert record["community_id"] in (1, 2, 3)
        members = np.flatnonzero(labels == record["community_id"])
        assert sorted(np.concatenate([record["seeds"], record["hidden"]]).tolist()) == members.tolist()
        assert len(record["seeds"]) == round(0.2 * len(members))
    # fewer eligible communities than requested: all of them
    records = synth.community_seed_sets(labels, 10, 10, 40, 0.2, np.random.default_rng(0))
    assert [r["community_id"] for r in records] == [1, 2, 3]
