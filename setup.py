from setuptools import setup, find_packages

setup(
    name="xai-ids-pro",
    version="1.0.0",
    description="Explainable AI Intrusion Detection System",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24,<2.0",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "pyarrow>=13.0",
        "tensorflow>=2.13",
        "shap>=0.43",
        "lime>=0.2.0.1",
        "pydantic>=1.10,<2.0",
        "PyYAML>=6.0",
        "fastapi>=0.104",
        "uvicorn[standard]>=0.24",
        "python-jose[cryptography]>=3.3",
        "passlib[bcrypt]>=1.7.4",
        "streamlit>=1.28",
        "matplotlib>=3.7",
        "plotly>=5.17",
        "rich>=13.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4", "pytest-cov>=4.1"],
    },
    entry_points={
        "console_scripts": [
            "xai-ids=main:main",
        ],
    },
)
