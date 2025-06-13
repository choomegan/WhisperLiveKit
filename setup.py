from setuptools import setup, find_packages
setup(
    name="whisperlivekit",
    version="0.2.1",
    description="Real-time, Fully Local Whisper's Speech-to-Text and Speaker Diarization",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Quentin Fuxa",
    url="https://github.com/QuentinFuxa/WhisperLiveKit",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "librosa",
        "soundfile",
        "faster-whisper",
        "uvicorn",
        "websockets",
        "grpcio",
        "grpcio-tools"
    ],
    extras_require={
        "diarization": ["diart"],
        "vac": ["torch"],
        "sentence": ["mosestokenizer", "wtpsplit"],
        "whisper": ["whisper"],
        "whisper-timestamped": ["whisper-timestamped"],
        "mlx-whisper": ["mlx-whisper"],
        "openai": ["openai"],
        "simulstreaming": [
            "torch",
            "tqdm",
            "tiktoken",
            "numpy<2.0.0",
            "triton>=2.0.0,<3;platform_machine==\"x86_64\" and sys_platform==\"linux\" or sys_platform==\"linux2\"",
        ],
    },
    package_data={
        'whisperlivekit': ['web/*.html'],
        'whisperlivekit.simul_whisper': ['dual_license_simulstreaming.md'],
        'whisperlivekit.simul_whisper.whisper.assets': ['*.tiktoken', '*.npz'],
    },
    entry_points={
        'console_scripts': [
            'whisperlivekit-server=whisperlivekit.basic_server:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.9",
)