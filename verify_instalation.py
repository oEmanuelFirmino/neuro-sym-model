import sys
import neurosym as ns

print(f"✅ Biblioteca importada com sucesso!")
print(f"📍 Localização no disco: {ns.__file__}")
print(f"🔢 Versão: {ns.__version__}")

try:
    from neurosym.tensor import Tensor

    t1 = Tensor([1.0, 2.0], requires_grad=True)
    t2 = Tensor([3.0, 4.0], requires_grad=True)
    soma = t1 + t2
    print(f"✅ Teste rápido de Tensor: {t1.data} + {t2.data} = {soma.data}")
except ImportError as e:
    print(f"❌ Erro ao importar submódulos: {e}")
    sys.exit(1)
