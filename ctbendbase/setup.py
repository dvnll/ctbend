from setuptools import setup
import os
import sys


if sys.version_info < (3, 4):
    sys.exit("Python < 3.4 is not supported.")


def get_version(version_tuple):
    return ".".join(map(str, version_tuple))


init = os.path.join(
        os.path.dirname(__file__), ".", "", "__init__.py")

version_line = list(
        filter(lambda l: l.startswith("VERSION"), open(init))
)[0]

PKG_VERSION = get_version(eval(version_line.split("=")[-1]))

description = "CTBend base package"

setup(name="ctbendbase",
      version=PKG_VERSION,
      description=description,
      license="MIT",
      install_requires=["numpy", "scipy"])
