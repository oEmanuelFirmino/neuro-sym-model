from .python_backend import PythonBackend

current_backend = PythonBackend()


def set_backend(name: str):
    global current_backend
    if name == "python":
        current_backend = PythonBackend()
    # backend NumPy will be added in the future
    # elif name == 'torch':
    #     from .numpy_backend import NumpyBackend
    #     current_backend = NumpyBackend()
    else:
        raise ValueError(
            f"Backend desconhecido: '{name}'. Opções disponíveis: ['python']"
        )


def get_backend():
    return current_backend
