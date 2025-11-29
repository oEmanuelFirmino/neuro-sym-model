import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


from src.neurosym.tensor import Tensor
from src.neurosym.tensor.backend import set_backend


def run_benchmark(backend_name: str, matrix_size: int):

    set_backend(backend_name)

    print("-" * 50)
    print(f"Executando Benchmark para o backend: '{backend_name}'")
    print(f"Tamanho da Matriz: {matrix_size}x{matrix_size}")
    print("-" * 50)

    data_a = [[float(i + j) for j in range(matrix_size)] for i in range(matrix_size)]
    data_b = [[float(i * j) for j in range(matrix_size)] for i in range(matrix_size)]

    start_time = time.time()

    a = Tensor(data_a, requires_grad=True)
    b = Tensor(data_b, requires_grad=True)

    d = a.dot(b)
    e = d.relu()
    f = e.transpose()
    loss = f.sum()

    loss.backward()

    end_time = time.time()

    total_time = end_time - start_time
    print(f"Resultado do Loss: {loss.data:.2f}")
    print(f"Tempo Total: {total_time:.4f} segundos\n")
    return total_time


def main():

    matrix_size = 100

    print("=" * 50)
    print("  INICIANDO TESTE DE PERFORMANCE DE BACKENDS")
    print("=" * 50)

    python_time = run_benchmark("python", matrix_size)

    numpy_time = run_benchmark("numpy", matrix_size)

    print("=" * 50)
    print("  RESULTADOS DA COMPARAÇÃO")
    print("=" * 50)
    print(f"Tempo com Backend Python: {python_time:.4f}s")
    print(f"Tempo com Backend NumPy:  {numpy_time:.4f}s")

    if numpy_time > 0:
        speedup = python_time / numpy_time
        print(f"\nGanho de Performance (Speedup): {speedup:.2f}x mais rápido!")
    else:
        print("\nNão foi possível calcular o ganho de performance (divisão por zero).")


if __name__ == "__main__":
    main()
