from shapely import geometry

class Canvas():

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.zones = {}
    
    def uniform_polygon(self, polygon):
        print(polygon)
        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        print(polygon)
        if len(polygon) < 4:
            raise Exception('Number of points in polygon is too small')
        
        for x, y in polygon:
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                raise Exception("Polygone's point is not in canvas")
        
        uniform_polygon = geometry.Polygon(polygon)
        if not uniform_polygon.is_valid:
            raise Exception('Invalid polygon shape')
        return uniform_polygon

    def set_zones(self, zones):
        for ind, polygons in zones.items():
            try:
                correct_polygons = [self.uniform_polygon(p) for p in polygons]
            except Exception as e:
                print(e)
                self.zones = {}
                break
            self.zones[ind] = correct_polygons

    def is_in_canvas(self, point):
        x, y = point
        return x >= 0 and y >= 0 and x < self.width and y < self.height

    def which_zone(self, point):
        def is_in(point, polygon):
            return polygon.contains(point)

        if not self.is_in_canvas(point):
            raise Exception(f'Point {point} is not in canvas')

        point = geometry.Point(*point)
        zones = []
        for ind, polygons in self.zones.items():
            if any([is_in(point, polygon) for polygon in polygons]):
                zones.append(ind)
        return zones
    
    def zone_crossing(self, vector):
        try:
            source = self.which_zone(vector[0])
            destination = self.which_zone(vector[1])
        except Exception as e:
            print(e)
            return
        return (source, destination)
        
        
c = Canvas(45, 15)
poly = {1:[[(0, 0), (7, 7), (0, 11)]]} 
c.set_zones(poly)
print(c.which_zone((7, 7)))
print(c.zone_crossing([(2, 5), (16, 8)]))