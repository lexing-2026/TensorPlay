"""Elastic rendezvous: backends, parameters, and error taxonomy."""
from collections.abc import Callable

from .api import (
    create_handler,
    get_registry,
    RendezvousClosedError,
    RendezvousConnectionError,
    RendezvousError,
    RendezvousExhaustedError,
    RendezvousGracefulExitError,
    RendezvousHandler,
    RendezvousHandlerRegistry,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStateError,
    RendezvousStoreInfo,
    RendezvousTimeout,
    RendezvousTimeoutError,
)
from .p10d_rendezvous_backend import P10dRendezvousBackend
from .c10d_rendezvous_backend import C10dRendezvousBackend
from .dynamic_rendezvous import DynamicRendezvousHandler, RendezvousSettings
from .static_tcp_rendezvous import StaticTCPRendezvous
from .utils import (
    _matches_machine_hostname,
    _parse_rendezvous_config,
    _PeriodicTimer,
    parse_rendezvous_endpoint,
)

__all__ = [
    "RendezvousError",
    "RendezvousClosedError",
    "RendezvousTimeoutError",
    "RendezvousConnectionError",
    "RendezvousStateError",
    "RendezvousGracefulExitError",
    "RendezvousExhaustedError",
    "RendezvousStoreInfo",
    "RendezvousInfo",
    "RendezvousHandler",
    "RendezvousParameters",
    "RendezvousHandlerRegistry",
    "RendezvousTimeout",
    "RendezvousSettings",
    "DynamicRendezvousHandler",
    "StaticTCPRendezvous",
    "P10dRendezvousBackend",
    "C10dRendezvousBackend",
    "create_handler",
    "get_registry",
    "parse_rendezvous_endpoint",
    "_parse_rendezvous_config",
    "_matches_machine_hostname",
    "_PeriodicTimer",
]
