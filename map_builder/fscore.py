import geopandas as gpd


def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def find_lines(standard, predicted):
    prdt_points = [prdt_point for prdt_line in predicted.geometry for prdt_point in prdt_line.coords]
    lines_found = 0
    for std_line in standard.geometry:
        points_found = 0
        for std_point in std_line.coords:
            for prdt_point in prdt_points:
                if distance(std_point, prdt_point) < 5:
                    points_found += 1
                    break
        std_line_point_count = len(std_line.coords)
        if points_found > std_line_point_count / 2:
            lines_found += 1
    return lines_found


def f_score(standard_file: str, neutral_file: str, predicted_file: str) -> float:
    """
    Compute f-score for given sets of paths.

    :param standard_file: shapefile with paths that should have been found
    :param neutral_file: shapefile with paths that are niether correct nor incorrect
    :param predicted_file: shapefile with predicted paths
    :return: f-score of prediction
    """
    standard = gpd.read_file(standard_file)
    neutral = gpd.read_file(neutral_file)
    predicted = gpd.read_file(predicted_file)

    standard_line_count = standard.shape[0]
    predicted_line_count = predicted.shape[0]
    lines_found = find_lines(standard, predicted)
    neutral_lines_count = find_lines(neutral, predicted)

    recall = lines_found / standard_line_count
    precision = lines_found / (predicted_line_count - neutral_lines_count)
    return recall * precision / (recall + precision)
