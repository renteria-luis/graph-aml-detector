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

## Bitcoin and the UTXO model

Bitcoin tracks money as UTXOs (unspent transaction outputs): discrete chunks of
coin of fixed value that an owner controls, like signed checks. To pay, you
create a transaction that consumes one or more UTXOs as inputs and produces new
outputs (the payment plus any change). A transaction is not a container of
money; it destroys input UTXOs and creates output UTXOs, and the coins live in
the outputs.

Example: A holds one 10 BTC UTXO and pays B 5 BTC. The transaction consumes the
whole 10 BTC UTXO and creates two outputs, 5 BTC to B and 5 BTC change to A.

This is why transactions form a directed graph: an edge runs from tx_a to tx_b
when tx_b spends an output of tx_a. A transaction can spend outputs from several
parents and fund several children, so the shapes that appear are one-to-many,
one-to-one, and many-to-one (consolidating two incoming payments into one
outgoing payment gives two parents and one child).

## What flagging a transaction means

Flagging is done by external chain-analysis firms, not by the Bitcoin protocol;
there is no authority inside Bitcoin. What happens depends on the state of the
transaction.

Unconfirmed (still in the mempool): nothing is final. The protocol has no fraud
mechanism, so a flag has no direct on-chain effect, but the transaction may
never get confirmed, and the owner's UTXOs stay spendable. Some
compliance-oriented mining pools screen sanctioned addresses, but that is not a
protocol feature.

Confirmed (in a block): immutable, and it cannot be reverted without breaking
consensus. Control happens off-chain instead. Chain-analysis services trace
UTXO flows forward and backward. When tainted funds eventually reach an
exchange, even after many hops and a valid final transaction, the exchange can
trace back to the flagged source and see something like "exposed to illicit
source". Exchanges can then freeze funds and decide whether to accept, block, or
review a deposit or withdrawal. The main users are exchanges and compliance
teams, and only rarely law enforcement.
