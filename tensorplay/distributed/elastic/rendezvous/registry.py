"""Registration of the built-in rendezvous backends."""
from .api import RendezvousHandlerRegistry, RendezvousParameters, RendezvousHandler
from .dynamic_rendezvous import create_handler as _create_core_handler
from .etcd_rendezvous import create_rdzv_handler as _create_etcd_handler
from .static_tcp_rendezvous import create_handler as _create_static_handler


def _register_default_handlers(registry: RendezvousHandlerRegistry) -> None:
    registry.register("static", _create_static_handler)
    registry.register("core", _create_core_handler)
    registry.register("p10d", _create_p10d_handler)
    registry.register("etcd", _create_etcd_handler)
    registry.register("etcd-v2", _create_etcd_handler)


def _create_p10d_handler(params: RendezvousParameters):
    from .dynamic_rendezvous import DynamicRendezvousHandler, RendezvousSettings
    from .p10d_rendezvous_backend import create_backend

    backend, store = create_backend(params)
    settings = RendezvousSettings(
        join=params.timeout.join,
        last_call=params.timeout.last_call,
        close=params.timeout.close,
        heartbeat=params.timeout.heartbeat,
        min_nodes=params.min_nodes,
        max_nodes=params.max_nodes,
    )
    return DynamicRendezvousHandler(
        backend=backend,
        settings=settings,
        local_addr=params.local_addr,
        node_rank=params.node_rank,
        run_id=params.run_id,
        store=store,
        backend_name=backend.name,
    )


def get_rendezvous_handler(params: RendezvousParameters) -> RendezvousHandler:
    """Create a handler for ``params.backend`` from the default registry."""
    from .api import create_handler

    return create_handler(params)
