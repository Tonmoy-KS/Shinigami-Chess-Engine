from setuptools import setup, find_packages

setup(
    name="shinigami-chess-engine",
    version="1.18.5",
    description="Shinigami Chess engine is a UX-focused chess engine in python for the goal of teaching beginner to intermediate developers the basics of modern chess engine architecture.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Tonmoy KS",
    author_email="your-email@example.com",
    url="https://github.com/Tonmoy-KS/Shinigami-Chess-Engine",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "python-chess",
        "numpy",
        "scipy",
        "openai",
        "torch",
        "pyttsx3",
        "deap"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Games/Entertainment :: Board Games",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "shinigami-chess=shinigami_chess_engine.engine:main"
        ]
    },
    include_package_data=True,
)