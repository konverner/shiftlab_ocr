# CLASS TO STORE INFORMATION ABOUT A CROP
class Crop:
    def __init__(self, points, img, label=None):
        """
        params
        ---
        points : list
            list of two pairs where each pair is (x,y) coordinates of the point of box
        img: PIL.Image
        label: string
        """
        self.points = points
        self.label = label
        self.img = img

    def __eq__(self, other):
        return self.mean_point == other.mean_point

    def __lt__(self, other):
        same_line = False
        y_mean = (self.points[1][1] + self.points[0][1]) / 2  # mean point for y coord

        if other.points[1][1] >= y_mean >= other.points[0][1]:  # check whether other is on the same line
            same_line = True
        if not same_line:
            if y_mean < other.points[0][1]:
                return True
            elif y_mean > other.points[1][1]:
                return False
        elif self.points[1][0] < other.points[1][0] and same_line:
            return True
        else:
            return False

    def __rt__(self, other):
        return self.mean_point > other.mean_point