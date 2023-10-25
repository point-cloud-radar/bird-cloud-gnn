# Bird Cloud GNN

[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/point-cloud-radar/bird-cloud-gnn)
[![github license badge](https://img.shields.io/github/license/point-cloud-radar/bird-cloud-gnn)](https://github.com/point-cloud-radar/bird-cloud-gnn)
[![RSD]( https://img.shields.io/badge/rsd-bird_cloud_gnn-blue)](https://research-software-directory.org/projects/bird-movements-with-meteorological-radar)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)
[![workflow scq badge](https://sonarcloud.io/api/project_badges/measure?project=point-cloud-radar_bird-cloud-gnn&metric=alert_status)](https://sonarcloud.io/dashboard?id=point-cloud-radar_bird-cloud-gnn)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=point-cloud-radar_bird-cloud-gnn&metric=coverage)](https://sonarcloud.io/dashboard?id=point-cloud-radar_bird-cloud-gnn)
[![read the docs badge](https://readthedocs.org/projects/bird-cloud-gnn/badge/?version=latest)](https://bird-cloud-gnn.readthedocs.io/en/latest/?badge=latest)
[![build](https://github.com/point-cloud-radar/bird-cloud-gnn/actions/workflows/build.yml/badge.svg)](https://github.com/point-cloud-radar/bird-cloud-gnn/actions/workflows/build.yml)
[![cffconvert](https://github.com/point-cloud-radar/bird-cloud-gnn/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/point-cloud-radar/bird-cloud-gnn/actions/workflows/cffconvert.yml)
[![sonarcloud](https://github.com/point-cloud-radar/bird-cloud-gnn/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/point-cloud-radar/bird-cloud-gnn/actions/workflows/sonarcloud.yml)
[![markdown-link-check](https://github.com/point-cloud-radar/bird-cloud-gnn/actions/workflows/markdown-link-check.yml/badge.svg)](https://github.com/point-cloud-radar/bird-cloud-gnn/actions/workflows/markdown-link-check.yml)



<!-- [![DOI](https://zenodo.org/badge/500818250.svg)](https://zenodo.org/badge/latestdoi/500818250) -->
<!-- [![docker hub badge](https://img.shields.io/static/v1?label=Docker%20Hub&message=mexca&color=blue&style=flat&logo=docker)](https://hub.docker.com/u/mexca) -->
<!-- [![docker build badge](https://img.shields.io/github/actions/workflow/status/mexca/mexca/docker.yml?label=Docker%20build&logo=docker)](https://github.com/mexca/mexca/actions/workflows/docker.yml) -->
[![black code style badge](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Synopsis

This software produces a graph representation derived from point cloud data, which is then used as input for a Graph Neural Network (GNN).This allows to increase the amount of data by the factor of 1000.

## Code example

## Motivation
In scenarios where labeled data is limited, there's a pressing need to expand the dataset effectively. One effective strategy involves altering the data's representation. In this context, we adopted such an approach by acquiring a graph representation from point cloud data. Depending on the chosen parameters, this transformation can augment the dataset by a factor of up to 1000. Subsequently, this graph representation is harnessed as input for Graph Neural Networks (GNNs). GNNs are highly sought after due to their innate ability to adeptly capture and leverage the inherent properties of graph-structured data. They excel in modeling intricate network relationships, autonomously acquiring informative features, and facilitating effective knowledge transfer.

## Installation

To install bird_cloud_gnn from GitHub repository, do:

```console
git clone https://github.com/point-cloud-radar/bird-cloud-gnn.git
cd bird-cloud-gnn
python3 -m pip install .
```

## Documentation

The documentation can be found on [Read the Docs](https://bird-cloud-gnn.readthedocs.io/en/latest/index.html).

## Contributing

If you want to contribute to the development of bird_cloud_gnn,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
