from dsplot.graph import Graph

graph = Graph(
    {0: [1, 4, 5], 1: [3, 4], 2: [1], 3: [2, 4], 4: [], 5: []},
    directed=True,
    edges={'01': 1, '04': 4, '05': 5, '13': 3, '14': 4, '21': 2, '32': 3, '34': 4},
)
graph.plot()