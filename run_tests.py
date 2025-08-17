#!/usr/bin/env python3
"""
Script simples para executar os testes do Tensor
Coloque este arquivo na raiz do projeto
"""

import sys
import os
from pathlib import Path

# Adicionar o diretório src ao path
project_root = Path(__file__).parent
src_path = project_root / "src"
tensor_path = src_path / "tensor"

sys.path.insert(0, str(src_path))
sys.path.insert(0, str(tensor_path))

try:
    from src.tensor.test_tensor import main

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    print(f"📁 Verificando estrutura de arquivos...")
    print(f"   Project root: {project_root}")
    print(f"   Src path: {src_path} (exists: {src_path.exists()})")
    print(f"   Tensor path: {tensor_path} (exists: {tensor_path.exists()})")

    tensor_py = tensor_path / "tensor.py"
    test_py = tensor_path / "test_tensor.py"
    print(f"   tensor.py: {tensor_py} (exists: {tensor_py.exists()})")
    print(f"   test_tensor.py: {test_py} (exists: {test_py.exists()})")

    print("\n💡 Soluções possíveis:")
    print("   1. Execute: cd src/tensor && python test_tensor.py")
    print("   2. Verifique se todos os arquivos estão nos locais corretos")
    print("   3. Certifique-se de que os arquivos __init__.py existem")

except Exception as e:
    print(f"❌ Erro durante execução: {e}")
    import traceback

    traceback.print_exc()
