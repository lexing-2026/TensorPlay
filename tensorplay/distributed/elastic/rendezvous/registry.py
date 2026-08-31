"""Registration of the built-in rendezvous backends."""
from .api import RendezvousHandlerRegistry, RendezvousParameters, RendezvousHandler
from .dynamic_rendezvous import create_handler as _create_core_handler
from .etcd_rendezvous import create_rdzv_handler as _create_etcd_handler
from .static_tcp_rendezvous import create_handler as _create_static_handler


def _register_default_handlers(registry: RendezvousHandlerRegistry) -> None:
    registry.register("static", _create_static_handler)
    registry.register("core", _create_core_handler)
    registry.register("etcd", _create_etcd_handler)
    registry.register("etcd-v2", _create_etcd_handler)


def get_rendezvous_handler(params: RendezvousParameters) -> RendezvousHandler:
    """Create a handler for ``params.backend`` from the default registry."""
    from .api import create_handler

    return create_handler(params)
