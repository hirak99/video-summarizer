"""Some algorithms to be used in process_graph management.

For the purpose of this module, a node is an integer and a graph is a mapping of
node to dependent nodes, dict[int, set[int]].
"""

import collections


def _get_dependencies(
    start_nodes: set[int], dependencies: dict[int, set[int]]
) -> set[int]:
    visited: set[int] = set()
    to_check = start_nodes

    while to_check:
        node = to_check.pop()
        visited.add(node)
        for dep in dependencies.get(node, set()):
            if dep not in visited:
                to_check.add(dep)

    return visited


def _topo_sort(dependencies: dict[int, set[int]]) -> list[int]:
    in_degree: dict[int, int] = collections.defaultdict(int)
    # Reverse-dependency graph.
    reverse_graph: dict[int, set[int]] = collections.defaultdict(set)

    # Get all nodes.
    nodes = set(dependencies.keys())
    for deps in dependencies.values():
        nodes.update(deps)

    # Compute in-degrees, and reverse-dependency graph.
    for node in nodes:
        in_degree[node] = 0

    for node, deps in dependencies.items():
        for dep in deps:
            reverse_graph[dep].add(node)
            in_degree[node] += 1

    # Standard algorithm - start with nodes of in-degree 0, pop and reduce
    # degrees of reverse-neighbors.
    queue = collections.deque([node for node in nodes if in_degree[node] == 0])
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in reverse_graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topo_order) != len(nodes):
        raise ValueError("Graph has at least one cycle. Topological sort not possible.")

    return topo_order


def topo_sort_subgraph(
    start_nodes: set[int], dependencies: dict[int, set[int]]
) -> list[int]:
    subgraph_nodes = _get_dependencies(start_nodes, dependencies)

    subgraph = {}
    for node in subgraph_nodes:
        subgraph[node] = dependencies.get(node, set()) & subgraph_nodes

    return _topo_sort(subgraph)
