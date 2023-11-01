"""Generation of fake point cloud radar data."""

from math import ceil
import numpy as np
import pandas as pd
from scipy.spatial import KDTree


# pylint: disable=too-many-arguments, too-many-locals
def generate_data(
    filename=None,
    num_points=2**13,
    min_range=50.0,
    max_range=300_000.0,
    azimuth_skip=2.0,
    elevations=np.array([0.3, 0.8, 1.2, 2, 2.8, 4.5, 6, 8, 10, 12, 15, 20, 25]),
    add_na=False,
    z_position=343.0,
    radius_influence=300.0,
):
    """Generate fake point cloud radar data

    The fake data containts polar coordinates `range`, `azimuth`, and `elevation`; cartesian
    coordinates `x`, `y`, `z`; random numerical features `useless_feature`, `feat1`, `feat2`,
    and `feat3`; and the target `class` with values 0 or 1.

    - `range` is generated like a exponential decay.
    - `azimuth` is generated uniformly in the interval [0.5, 365.5).
    - `elevation` is taken from an input array.
    - `x`, `y`, and `z` are converted from these polar values.
    - `useless_feature` is taken from a Normal(0, 1) distribution.
    - `feat1`, `feat2`, and `feat3` are randomly constructed to remain in the interval [0, 1].


     The columns are `range`, `azimuth` and `elevation` for the polar coordinates,
    `x`, `y`, and `z` for the cartesian coordinates, `useless_feature`, `feat1`, `feat2`,
    `feat3` for the numerical features, and `class` for the target.

    Args:
        filename (str, optional): Filename to save the data. Use None to ignore.
        num_points (int, optional): Number of points. Defaults to 2**13.
        max_range (float, optional): Maximum generated range. Defaults to 300_000.0.
        azimuth_skip (float, optional): Size between azimuth values. Defaults to 2.0.
        elevations (array of floats, optional): List of elevations. Defaults to
            np.array([0.3, 0.8, 1.2, 2, 2.8, 4.5, 6, 8, 10, 12, 15, 20, 25]).
        add_na (bool, optional): Whether to add missing data. Defaults to False.
        z_position (float, optional): z position of the radar. Used for offsetting z after
            converting from polar to cartesian. Defaults to 343.0.
        radius_influence (float, optional): Radius used to compute the number of neighbours
            used internally for predicting the target class. Defaults to 300.0.

    Returns:
        pandas.DataFrames: Generated data. It is also saved to `filename`
        if that argument is passed.
    """

    point_cloud = pd.DataFrame(
        {
            "range": min_range
            + np.exp(-5 * np.random.rand(num_points)) * (max_range - min_range),
            "azimuth": np.tile(
                np.arange(0, 360, azimuth_skip) + 0.5,
                ceil(num_points / 360 * azimuth_skip),
            )[0:num_points],
            "elevation": sorted(
                np.tile(elevations, ceil(num_points / len(elevations)))[0:num_points]
            ),
            "useless_feature": np.random.randn(num_points),
        }
    )

    c_elevation, s_elevation = np.cos(np.pi / 180 * point_cloud.elevation), np.sin(
        np.pi / 180 * point_cloud.elevation
    )
    c_azimuth, s_azimuth = np.cos(np.pi / 180 * point_cloud.azimuth), np.sin(
        np.pi / 180 * point_cloud.azimuth
    )
    point_cloud["x"] = point_cloud.range * c_elevation * s_azimuth
    point_cloud["y"] = point_cloud.range * c_elevation * c_azimuth
    point_cloud["z"] = point_cloud.range * s_elevation + z_position

    xyz = ["x", "y", "z"]
    tree = KDTree(point_cloud.loc[:, xyz])

    # Pretty much random choices below
    def sigmoid(value):
        return 1 / (1 + np.exp(-value))

    def to01(value):
        return (value - np.min(value)) / (np.max(value) - np.min(value))

    point_cloud["feat1"] = sigmoid(
        (point_cloud.x + point_cloud.y - point_cloud.z) / max_range
    )
    point_cloud["feat2"] = sigmoid(
        (point_cloud.z - point_cloud.z.mean()) / point_cloud.z.std()
    )
    point_cloud["feat3"] = (
        1
        + np.cos(
            np.exp(
                -1
                + 2
                * to01(
                    -0.3 * (point_cloud.x - 1) ** 2 - 0.2 * (point_cloud.y + 0.3) ** 2
                )
            )
        )
    ) / 2
    hidden1 = (point_cloud.feat1 + point_cloud.feat2**2) / point_cloud.feat3
    hidden1 = (hidden1 - np.mean(hidden1)) / np.std(hidden1)
    hidden2 = np.log(1 + point_cloud.feat1**2) - np.sin(4 * np.pi * point_cloud.feat2)
    hidden2 = (hidden2 - np.mean(hidden2)) / np.std(hidden2)

    neighbours = point_cloud.apply(
        lambda row: len(tree.query_ball_point(row[["x", "y", "z"]], radius_influence)),
        axis=1,
    )
    aux = (neighbours > 5).astype("int32")
    point_cloud["class"] = np.round(
        sigmoid(0.5 * hidden1 + 0.2 * hidden2 + 3 * aux)
        + np.random.randn(num_points) * 0.1
    )

    if add_na:
        point_cloud.loc[
            np.random.randint(0, num_points, num_points // 100), "feat2"
        ] = None
        point_cloud.loc[
            np.random.randint(0, num_points, num_points // 20), "feat3"
        ] = None

    if filename is not None:
        point_cloud.to_csv(filename, index=None)

    return point_cloud
