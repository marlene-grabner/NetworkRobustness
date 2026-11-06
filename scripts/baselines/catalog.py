##################################################
# Catalog of the extended baseline set.
# Each entry describes where a network is fetched
# from and which domain it belongs to.
#
# source = "netzschleuder": dataset/net refer to https://networks.skewed.de/net/<dataset>
# source = "snap":          file refers to https://snap.stanford.edu/data/<file>
# directed: whether the ORIGINAL network is directed (it is symmetrized on conversion)
#
# Selection aims for diversity: at most one collaboration network and one
# protein interaction network, so that no single network type dominates.
##################################################

NETWORKS = [
    # ---------------- Social ----------------
    {"key": "facebook_combined", "domain": "social", "subdomain": "online_friendship", "source": "snap", "file": "facebook_combined.txt.gz", "directed": False},
    {"key": "lastfm_asia", "domain": "social", "subdomain": "online_friendship", "source": "netzschleuder", "dataset": "lastfm_asia", "net": "lastfm_asia", "directed": False},
    {"key": "pgp_2004", "domain": "social", "subdomain": "online_trust", "source": "netzschleuder", "dataset": "pgp_2004", "net": "pgp_2004", "directed": False},
    {"key": "uni_email", "domain": "social", "subdomain": "communication", "source": "netzschleuder", "dataset": "uni_email", "net": "uni_email", "directed": True},
    {"key": "ca_condmat", "domain": "social", "subdomain": "collaboration", "source": "snap", "file": "ca-CondMat.txt.gz", "directed": False},
    {"key": "trumpworld", "domain": "social", "subdomain": "political", "source": "netzschleuder", "dataset": "trumpworld", "net": "trumpworld", "directed": False},
    {"key": "tennis_atp", "domain": "social", "subdomain": "sports", "source": "netzschleuder", "dataset": "tennis_atp", "net": "wins", "directed": True},
    {"key": "marvel_universe", "domain": "social", "subdomain": "fictional", "source": "netzschleuder", "dataset": "marvel_universe", "net": "marvel_universe", "directed": False},
    {"key": "ragusa_genealogy", "domain": "social", "subdomain": "genealogy", "source": "netzschleuder", "dataset": "ragusa_genealogy", "net": "ragusa", "directed": True},
    # ---------------- Biological ----------------
    {"key": "yeast_biogrid_2007", "domain": "biological", "subdomain": "protein_interactions", "source": "netzschleuder", "dataset": "yeast_biogrid_2007", "net": "yeast_biogrid_2007", "directed": False},
    {"key": "lit_bm", "domain": "biological", "subdomain": "protein_interactions_literature", "source": "netzschleuder", "dataset": "huri", "net": "Lit-BM", "directed": False},
    {"key": "yeast_regulatory", "domain": "biological", "subdomain": "gene_regulation", "source": "netzschleuder", "dataset": "gerstein_regulatory", "net": "yeast", "directed": True},
    {"key": "arabidopsis_metabolic", "domain": "biological", "subdomain": "metabolic", "source": "netzschleuder", "dataset": "kegg_metabolic", "net": "ath", "directed": False},
    {"key": "human_connectome", "domain": "biological", "subdomain": "connectome", "source": "netzschleuder", "dataset": "human_connectomes_multiscale", "net": "hcp_00_layer0", "directed": False},
    {"key": "plant_hypocotyl", "domain": "biological", "subdomain": "cell_adjacency", "source": "netzschleuder", "dataset": "plant_hypocotyl", "net": "ara_col_1", "directed": False},
    {"key": "fungal", "domain": "biological", "subdomain": "fungal_spatial", "source": "netzschleuder", "dataset": "fungal_networks", "net": "Ag_M_I+4R_U_N_42d_1", "directed": False},
    {"key": "host_pathogen", "domain": "biological", "subdomain": "ecological", "source": "netzschleuder", "dataset": "host_pathogen_eid2", "net": "species", "directed": True},
    # ---------------- Infrastructure (energy, water, transport) ----------------
    {"key": "gridkit_europe", "domain": "infrastructure", "subdomain": "power_grid", "source": "netzschleuder", "dataset": "gridkit", "net": "europe", "directed": False},
    {"key": "scigrid_gas", "domain": "infrastructure", "subdomain": "gas_pipelines", "source": "netzschleuder", "dataset": "scigrid_gas", "net": "v1.0.0", "directed": False},
    {"key": "water_bwsn2", "domain": "infrastructure", "subdomain": "water_distribution", "source": "netzschleuder", "dataset": "water_distribution_exeter", "net": "bwsn_network_2", "directed": True},
    {"key": "us_roads_dc", "domain": "infrastructure", "subdomain": "roads", "source": "netzschleuder", "dataset": "us_roads", "net": "DC", "directed": False},
    {"key": "chicago_road", "domain": "infrastructure", "subdomain": "roads", "source": "netzschleuder", "dataset": "chicago_road", "net": "chicago_road", "directed": True},
    {"key": "railway_italy", "domain": "infrastructure", "subdomain": "railway", "source": "netzschleuder", "dataset": "railway_delays", "net": "italy", "directed": False},
    {"key": "london_transport", "domain": "infrastructure", "subdomain": "public_transport", "source": "netzschleuder", "dataset": "gb_public_transport", "net": "london", "directed": False},
    {"key": "openflights", "domain": "infrastructure", "subdomain": "airports", "source": "netzschleuder", "dataset": "openflights", "net": "openflights", "directed": True},
    {"key": "us_airports", "domain": "infrastructure", "subdomain": "airports", "source": "netzschleuder", "dataset": "us_airports", "net": "bts_2010", "directed": True},
    # ---------------- Technological (internet, software, circuits) ----------------
    {"key": "internet_as", "domain": "technological", "subdomain": "internet_as", "source": "netzschleuder", "dataset": "internet_as", "net": "internet_as", "directed": False},
    {"key": "routers_1239", "domain": "technological", "subdomain": "internet_routers", "source": "netzschleuder", "dataset": "rocketfuel", "net": "routers_1239", "directed": False},
    {"key": "gnutella08", "domain": "technological", "subdomain": "peer_to_peer", "source": "snap", "file": "p2p-Gnutella08.txt.gz", "directed": True},
    {"key": "linux_callgraph", "domain": "technological", "subdomain": "software", "source": "netzschleuder", "dataset": "software_callgraphs", "net": "linux", "directed": True},
    {"key": "circuit_s9234", "domain": "technological", "subdomain": "electronic_circuit", "source": "netzschleuder", "dataset": "iscas89_circuits", "net": "s9234", "directed": True},
    # ---------------- Informational (citation, web, language, knowledge) ----------------
    {"key": "cora", "domain": "informational", "subdomain": "citation", "source": "netzschleuder", "dataset": "cora", "net": "cora", "directed": True},
    {"key": "dblp_cite", "domain": "informational", "subdomain": "citation", "source": "netzschleuder", "dataset": "dblp_cite", "net": "2017", "directed": True},
    {"key": "cjeu_citations", "domain": "informational", "subdomain": "legal_citation", "source": "netzschleuder", "dataset": "cjeu_citations", "net": "cjeu_citations", "directed": True},
    {"key": "polblogs", "domain": "informational", "subdomain": "web", "source": "netzschleuder", "dataset": "polblogs", "net": "polblogs", "directed": True},
    {"key": "wiki_chameleon", "domain": "informational", "subdomain": "web", "source": "netzschleuder", "dataset": "wikipedia_article_networks", "net": "chameleon", "directed": False},
    {"key": "word_adjacency_darwin", "domain": "informational", "subdomain": "language", "source": "netzschleuder", "dataset": "word_adjacency", "net": "darwin", "directed": True},
    {"key": "odlis", "domain": "informational", "subdomain": "language", "source": "netzschleuder", "dataset": "odlis", "net": "odlis", "directed": True},
    {"key": "codex_kg", "domain": "informational", "subdomain": "knowledge_graph", "source": "netzschleuder", "dataset": "codex_kg", "net": "codex_s", "directed": True},
    # ---------------- Economic ----------------
    {"key": "lightning_2019", "domain": "economic", "subdomain": "payment_channels", "source": "netzschleuder", "dataset": "lightning_network", "net": "2019_11_18", "directed": True},
    {"key": "bitcoin_alpha", "domain": "economic", "subdomain": "trade_trust", "source": "netzschleuder", "dataset": "bitcoin_alpha", "net": "bitcoin_alpha", "directed": True},
    {"key": "amazon_ratings", "domain": "economic", "subdomain": "co_purchase", "source": "netzschleuder", "dataset": "heterophilous_graphs", "net": "amazon_ratings", "directed": False},
    {"key": "faculty_hiring", "domain": "economic", "subdomain": "labor_market", "source": "netzschleuder", "dataset": "faculty_hiring_us", "net": "academia", "directed": True},
    {"key": "us_migration", "domain": "economic", "subdomain": "migration", "source": "netzschleuder", "dataset": "us_county_flows", "net": "migration_2014_2015", "directed": True},
]
