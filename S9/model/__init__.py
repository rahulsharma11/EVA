import importlib
import torchvision
from model.model import cfar10_model

def get_model(name, **model_args):
    """
    Get a model by name.
    """
    try:
        print(name['name'])
        # model = importlib.import_module('.'+name['name'],'model')
        model = cfar10_model(**model_args)
        return model

    except ImportError:
        raise ValueError('Model {} not found.'.format(name['name']))
        ModelType = getattr(torchvision.models, name)
        return ModelType(**model_args)
        