import numpy as np

class Point(object):
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        self.change = 0
        self.cluster = None
        self.points = 0 # if this point is center, then self.points > 0

    def dist(self, point):
        return (self.x1-point.x1)**2 + (self.x2-point.x2)**2
    
    def assign(self, cluster):
        self.cluster = cluster
    
    def __eq__(self, point):
        return self.x1==point.x1 and self.x2==point.x2


def kmeans(centers, points):
    new_centers = [Point(0.0,0.0) for i in range(len(centers))]

    # step 1, assign 
    for point in points:
        cluster = np.argmin([point.dist(c) for c in centers])
        #print(point.x1,point.x2,cluster)
        if point.cluster != cluster:
            point.change += 1
            point.assign(cluster)
        new_centers[cluster].x1 += point.x1
        new_centers[cluster].x2 += point.x2
        new_centers[cluster].points += 1
    for c in new_centers:
        c.x1 /= c.points
    if new_centers==centers:
        print([(c.x1,c.x2) for c in new_centers])
        print([(c.x1,c.x2) for c in centers])
        return
    else: kmeans(new_centers, points)


centers = [Point(2.0,2.0), Point(-2.0,-2.0)]
points = [Point(-1.88,2.05), Point(-0.71,0.42), Point(2.41,-0.67), Point(1.85,-3.80), Point(-3.69,-1.33)]
kmeans(centers,points)

for i,p in enumerate(points):
    print('Point {}: change {} centers.'.format(i+1, p.change))


