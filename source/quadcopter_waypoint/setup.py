"""Installation script for the quadcopter_waypoint Isaac Lab external project."""

from setuptools import find_packages, setup

setup(
    name="quadcopter_waypoint",
    version="0.1.0",
    description="External Isaac Lab quadcopter waypoint RL task.",
    author="Ma Yingjie",
    packages=find_packages(),
    # Apply to every discovered package so each task's agents/*.yaml survives wheel builds.
    package_data={"": ["*.yaml"]},
    include_package_data=True,
    install_requires=["psutil"],
    python_requires=">=3.10",
    zip_safe=False,
)
