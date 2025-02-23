from .LiberoEngine import build_libero_processor, build_libero_dataloader, build_libero_agent, build_libero_engine
from .LiberoEngine import eval_libero

ENGINE_REPO = {
    'build_libero_engine': build_libero_engine
}

def create_engine(engine_name, **kwargs):
    return ENGINE_REPO[engine_name](**kwargs)