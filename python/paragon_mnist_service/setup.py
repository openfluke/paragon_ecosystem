from setuptools import setup, find_packages

setup(
    name="paragon_mnist_service",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pillow",
        "paragon_py",
    ],
    entry_points={
        "console_scripts": [
            "paragon-mnist-service=paragon_mnist_service.server:main",
        ],
    },
    author="OpenFluke",
    description="Dual-backend (CPU/GPU) Paragon MNIST microservice with hosted images",
    license="Apache-2.0",
    python_requires=">=3.9",
)
