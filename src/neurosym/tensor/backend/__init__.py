from .python_backend import PythonBackend

current_backend = PythonBackend()


def set_backend(name: str):

    global current_backend
    if name == "python":
        print("INFO: Usando backend 'python'.")
        current_backend = PythonBackend()
    elif name == "numpy":
        try:
            from .numpy_backend import NumpyBackend

            print("INFO: Usando backend 'numpy'.")
            current_backend = NumpyBackend()
        except ImportError:
            raise ImportError(
                "O backend 'numpy' não está disponível. Por favor, instale a biblioteca com 'pip install numpy'."
            )
    else:
        raise ValueError(
            f"Backend desconhecido: '{name}'. Opções disponíveis: ['python', 'numpy']"
        )


def get_backend():
    return current_backend
