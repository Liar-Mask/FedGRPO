import sys
import transformers
import accelerate
import trl

print(f"Python Executable Path: {sys.executable}")
print("-" * 50)
print(f"Transformers version: {transformers.__version__}")
print(f"Transformers library location: {transformers.__file__}")
print("-" * 50)
print(f"Accelerate version: {accelerate.__version__}")
print(f"Accelerate library location: {accelerate.__file__}")
print("-" * 50)
print(f"TRL version: {trl.__version__}")
print(f"TRL library location: {trl.__file__}")