"""Elastic agent and worker failover contract.

The elastic package runs user code across a set of nodes with:

* rendezvous-based formation of the worker world (``rendezvous``);
* a local agent per node that starts, monitors, restarts, and tears down
  workers (``agent``);
* process management with log redirection and failure capture
  (``multiprocessing``);
* pluggable events, metrics, and deadline timers (``events``, ``metrics``,
  ``timer``).

Agents do not synchronize restart counters among themselves; a node that
exhausts its own ``max_restarts`` closes the rendezvous, which terminates
in-flight workers on peer agents. Scale-up is detected through
``RendezvousHandler.num_nodes_waiting`` and triggers a re-rendezvous
without consuming a restart.
"""
