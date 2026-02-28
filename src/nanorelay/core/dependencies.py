from functools import lru_cache
from nanorelay.relay.dispatcher import Dispatcher

@lru_cache
def get_dispatcher() -> Dispatcher:
    return Dispatcher()
