from pprint import pprint


def rectangles_collide(actual_position, calculated_position):
    """
    Detects if the two given boxes collide
    """
    x_actual = actual_position.get("x", None)
    y_actual = actual_position.get("y", None)
    w_actual = actual_position.get("width", None)
    h_actual = actual_position.get("height", None)
    
    x_calculated = calculated_position.get("x", None)
    y_calculated = calculated_position.get("y", None)
    w_calculated = calculated_position.get("width", None)
    h_calculated =  calculated_position.get("height", None)

    #conditions
    x_a = (x_actual <= x_calculated) and (x_calculated < (x_actual + w_actual))
    x_b = (x_calculated <= x_actual) and (x_actual < (x_calculated + w_calculated))
    y_a = (y_actual <= y_calculated) and (y_calculated < (y_actual + w_actual))
    y_b = (y_calculated <= y_actual) and (y_actual < (y_calculated + w_calculated))

    if (((x_a) or (x_b)) and ((y_a) or (y_b))):
        return True


def calc_intersection(actual_position, calculated_position):
    """
    Calculates the Coordinates of the intersection Area.
    """
    x_actual = actual_position.get("x", None)
    y_actual = actual_position.get("y", None)
    w_actual = actual_position.get("width", None)
    h_actual = actual_position.get("height", None)
    
    x_calculated = calculated_position.get("x", None)
    y_calculated = calculated_position.get("y", None)
    w_calculated = calculated_position.get("width", None)
    h_calculated =  calculated_position.get("height", None)

    x1 = x_actual if(x_actual > x_calculated) else x_calculated
    y1 = y_actual if(y_actual > y_calculated) else y_calculated

    x2 = (x_actual + w_actual - 1) if((x_actual + w_actual - 1) < (x_calculated + w_calculated -1)) else (x_calculated + w_calculated - 1)

    y3 = (y_actual + h_actual -1) if((y_actual + h_actual -1) < (y_calculated + h_calculated -1)) else (y_calculated + h_calculated - 1)

    x = x1
    y = y1
    w = x2 - x1 + 1
    h = y3 - y1 + 1

    return {"x": x, "y": y, "width": w, "height": h}

def pixel_sum(position):
    """
    Sums up the total count of pixels in the area limited by the coordinate points in position.
    """
    if (position != None):
        return position.get("width", 0) * position.get("height", 0)
    else:
        return 0


def intersection_union(actual_position, calculated_position):
    """
    returns the values for the intersection and the union
    """
    intersection_coordinate = calc_intersection(actual_position, calculated_position)
    intersec = pixel_sum(intersection_coordinate)
    union = pixel_sum(actual_position) + pixel_sum(calculated_position) - intersec
    return intersec, union
