# Concepts

Working notes for the AML graph detector. One section per concept as the
project advances.

## The graph in this project

In the Elliptic dataset, a node is a Bitcoin transaction with 166 features, and
a directed edge is a flow of funds from one transaction to another. Some nodes
carry a licit or illicit label. The node features are an ordinary matrix (one
row per node), which a normal network handles fine. The hard part is the
connectivity: who connects to whom.

## The core problem: permutation invariance

The obvious way to encode connectivity is the adjacency matrix (N by N, a 1 at
(i, j) for an edge i to j). The flaw: the same graph has N! valid adjacency
matrices, one per node numbering. Renumber the nodes and the matrix changes
while the graph stays identical.

A plain MLP breaks on this. Its weights are bound to fixed positions in the
flattened input, so each numbering looks like a different input and produces a
different prediction for the same graph. The MLP has no way to tell that two
orderings are the same graph.

What we need is an operation whose output does not depend on node order:
permutation invariance at the graph level, equivariance at the node level. This
is the relational inductive bias of graphs, the analogue of translation
invariance for images.

## Consequences seen in the code

1. We use an adjacency list, not the matrix. In PyG this is `edge_index`, a
   2 by E matrix of [source, target] pairs. Sparse and order-free.
2. Each node aggregates its neighbors with an order-invariant operation (sum,
   mean, or max). That aggregation is the heart of message passing.

## AML connection

The suspicious pattern (a node receiving from many sources and sending to many
destinations) is a property of the structure, not of the neighbors' IDs. The
model must read the pattern of connections, not the labels. That is what
permutation invariance formalises.