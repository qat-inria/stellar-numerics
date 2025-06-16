from stellar.gaussian import check_gaussian_displacement


class TimeGaussianDisplacement:
    params = ([-5, 0, 5], [-5, 0, 5])

    def time_gaussian_displacement(self, x: float, y: float) -> None:
        check_gaussian_displacement(x, y)
