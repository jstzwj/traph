#-*- coding:utf-8 -*-

from setuptools import setup, find_packages

setup(
    name = "pytraph",
    version = "0.0.1",
    keywords = ("pip", "deep learning"),
    description = "Deep learning framework",
    long_description = "Deep learning framework",
    license = "MIT Licence",

    url = "https://github.com/toyteam/traph.git",
    author = "JunWang",
    author_email = "jstzwj@aliyun.com",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires = ["numpy"]
)