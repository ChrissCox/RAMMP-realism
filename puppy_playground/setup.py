from setuptools import setup

package_name = "puppy_playground"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    package_data={package_name: ["data/*.json", "web/*.html"]},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/playground.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            f"{package_name}_node = {package_name}.engine:main",
            f"{package_name}_ui = {package_name}.ui_node:main",
        ]
    },
)
