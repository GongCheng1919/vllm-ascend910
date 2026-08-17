from setuptools import setup, find_packages

setup(
    name="vllm-midgroup-plugin",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "vllm.general_plugins": [
            "midgroup_w4a8 = vllm_midgroup_plugin:register",
        ],
    },
)
