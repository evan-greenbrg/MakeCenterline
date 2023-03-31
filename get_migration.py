import math
import pandas
import pickle
import codecs
import os
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from shapely import geometry
from shapely import errors
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import Point 
from shapely.geometry import MultiPoint 
from shapely.geometry import GeometryCollection
from shapely import affinity
from matplotlib.widgets import Button
from scipy import spatial
from pyproj import Proj, transform


class PointPicker(object):
    text_template = 'x: %0.2f\ny: %0.2f'
    x, y = 0.0, 0.0
    xoffset, yoffset = -20, 20
    text_template = 'x: %0.2f\ny: %0.2f'

    def __init__(self, ax):
        self.ax = ax
        self.mouseX = []
        self.mouseY = []
        self.events = []
        self.cutoffsI = []
        self.points = []
        self.idx = 0
        self.p = []

    def clear(self, event):
        self.events = []
        self.mouseX = []
        self.mouseY = []
        self.cutoffsI = []

        for p in self.points:
            p.remove()
        self.p = []
        event.canvas.draw()

        print('Cleared')

    def next(self, event):
        self.idx = 0
        for p in self.points:
            p.remove()
        self.p = []

    def done(self, event):
        plt.close('all')

    def __call__(self, event):
        self.event = event
        self.events.append(event)
        self.x, self.y = event.mouseevent.xdata, event.mouseevent.ydata
        print(self.x, self.y)
        print(self.idx)
        if (self.x is not None) and (self.idx == 1):
            self.mouseX.append(self.x)
            self.mouseY.append(self.y)

            self.p.append(self.ax.scatter(self.x, self.y, color='black'))
            self.cutoff[1, 0] = self.x
            self.cutoff[1, 1] = self.y
            self.cutoffsI.append(self.cutoff)
            print('First Point')

            plt.scatter(self.x, self.y)
            self.idx += 1

        if self.idx == 0:
            self.cutoff = np.array([[self.x, self.y], [0, 0]])
            self.p.append(self.ax.scatter(self.x, self.y, color='black'))
            self.idx += 1
            print('Second Point')
        event.canvas.draw()


class Centerline:
    def __init__(self, mask, crs, transform):
        self.mask = mask
        self.crs = crs
        self.transform = transform
        

    def find_image_endpoints(self, endpoints, es):
        es = [v for v in es] 
        riv_end = np.empty([2,2])
        for idx, v in enumerate(es):
            if v == 'N':
                i = np.where(endpoints[:,1] == endpoints[:,1].min())[0][0]
            elif v == 'E':
                i = np.where(endpoints[:,0] == endpoints[:,0].max())[0][0]
            elif v == 'S':
                i = np.where(endpoints[:,1] == endpoints[:,1].max())[0][0]
            elif v == 'W':
                i = np.where(endpoints[:,0] == endpoints[:,0].min())[0][0]
            riv_end[idx, :] = endpoints[i,:]

        return riv_end


    def get_largest(self):
        labels = measure.label(self.mask)
         # assume at least 1 CC
        assert(labels.max() != 0)

        # Find largest connected component
        bins = np.bincount(labels.flat)[1:] 
        self.mask = labels == np.argmax(np.bincount(labels.flat)[1:])+1


    def find_intersections(self):
        rr, cc = np.where(self.mask)

        rows = []
        cols = []
        # Get neighboring pixels
        for r, c in zip(rr, cc):
            window = self.mask[r-1:r+2, c-1:c+2]

            if len(window[window]) > 3:
                rows.append(r)
                cols.append(c)

        return np.array([cols, rows]).transpose()


    def find_all_endpoints(self):
        rr, cc = np.where(self.mask)

        rows = []
        cols = []
        # Get neighboring pixels
        for r, c in zip(rr, cc):
            window = self.mask[r-1:r+2, c-1:c+2]

            if len(window[window]) < 3:
                rows.append(r)
                cols.append(c)

        return np.array([cols, rows]).transpose()


    def remove_small_segments(self, intersections, endpoints, thresh):
        tree = spatial.KDTree(intersections)
        costs = np.where(self.mask, 1, 1000)
        removed = 0
        for point in endpoints:
            distance, i = tree.query(point)
            path, dist = graph.route_through_array(
                costs, 
                start=(point[1], point[0]),
                end=(intersections[i][1], intersections[i][0]),
                fully_connected=True
            )

            path = np.array(path)
            if dist < thresh:
                self.mask[path[:,0], path[:,1]] = False
                removed += 1
            else:
                continue
            
        self.mask[intersections[:,1], intersections[:,0]] = True

        return removed

    def clean_centerline(self, es, thresh=10000):
        removed = 999
        self.get_largest()
        endpoints = self.find_all_endpoints()
        # Find the terminal endpoints
        river_endpoints = self.find_image_endpoints(endpoints, es)
        while removed > 2:
            # Find the all endpoints in the centerline
            endpoints = self.find_all_endpoints()
            # Add an intersection
            for end in river_endpoints:
                self.mask[
                    int(end[1]-1):int(end[1]+2),
                    int(end[0]
                )] = 1
                self.mask[
                    int(end[1]), 
                    int(end[0]-1):int(end[0]+2)
                ] = 1

            # Find all intersections
            intersections = self.find_intersections()

            # Remove all the small bits
            removed = self.remove_small_segments(
                intersections, 
                endpoints,
                thresh
            )
            print(removed)

        # Remove the fake intersection created at the river ends
        for end in river_endpoints:
            self.mask[
                int(end[1]-1):int(end[1]+2),
                int(end[0]
            )] = 0
            self.mask[
                int(end[1]), 
                int(end[0]-1):int(end[0]+2)
            ] = 0

    def get_idx(self):
        centerline.idx = np.array(np.where(centerline.mask)).T

    def get_xy(self):
        centerline.xy = np.array(rasterio.transform.xy(
            centerline.transform,
            centerline.idx[:, 0], 
            centerline.idx[:, 1], 
        )).T

    def get_graph(self):

        start = 0
        end = len(self.xy)-1
        tmp = [tuple(i) for i in self.xy]

        G = nx.Graph()
        H = nx.Graph()
        for idx, row in enumerate(tmp):
            G.add_node(idx, pos=row)
            H.add_node(idx, pos=row)

        # Add all edges
        for idx, nodeA in enumerate(tmp):
            for jdx, nodeB in enumerate(tmp):
                if idx == jdx:
                    continue
                else:
                    length = np.linalg.norm(np.array(nodeA) - np.array(nodeB))
                    G.add_edge(idx, jdx, length=length)

        # Reduce number of edges so each node only has two edges
        for node in G.nodes():
            # Get all edge lengths 
            edge_lengths = np.empty((len(G.edges(node)),))
            edges = np.array(list(G.edges(node)))
            for idx, edge in enumerate(edges):
                edge_lengths[idx] = G.get_edge_data(*edge)['length']

            # Only select the two smallest lengths
            if (node == start) or (node == end):
                ks = np.argpartition(edge_lengths, 2)[:1]
            else:
                ks = np.argsort(edge_lengths)[:2]

            use_edges = [tuple(i) for i in edges[ks]]

            # Add the filtered edges to the H network
            for edge in use_edges:
                length = G.get_edge_data(*edge)['length']
                H.add_edge(*edge, length=length)

        self.graph = JoinComponents(H, G)


def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))

    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)

    return ii, jj


def intersection(x1, y1, x2, y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.

usage:
x,y=intersection(x1,y1,x2,y2)

    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)

    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)

    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()

    """
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T

    return xy0[:, 0], xy0[:, 1]


def splitIntoChunks(I, xy1, xy2):
    """
    Splits centerlines into chunks that do not include channel cutoffs

    Inputs:
    cutoffs: indexes of cutoffs
    xy1: x-y coordinates at t1
    xy2: x-y coordinates at t2

    returns:
    segments: non-cutoff channel segments 
    """
    # Split centerlines into chunks that don't include cutoffs
    curindex1 = 0
    curindex2 = 0
    segments1 = []
    segments2 = []
    for idx, ind in enumerate(I):
        segments1.append(xy1[curindex1:int(ind[0])])
        segments2.append(xy2[curindex2:int(ind[1])])

        curindex1 = int(ind[0])
        curindex2 = int(ind[1])

        if idx != len(I)-1:
            nextindex1 = int(I[idx+1][0])
            nextindex2 = int(I[idx+1][1])

            segments1.append(xy1[curindex1:nextindex1])
            segments2.append(xy2[curindex2:nextindex2])

            curindex1 = nextindex1
            curindex2 = nextindex2

        else:
            segments1.append(xy1[curindex1:])
            segments2.append(xy2[curindex2:])

    return segments1, segments2


def findMigratedArea(xy1, xy2, I):
    """
    Finds the migrated area between two centerlines
    Inputs:

    xy1segs: list of t1 centerline segments (not cutoffs)
    xy2segs: list of t2 centerline segments

    Outputs:

    polygons: list of segment polygons of migrated area
    areas: list of polygon areas
    """
    xy1segs, xy2segs = splitIntoChunks(I, xy1, xy2)
    polygons = []
    for xy1seg, xy2seg in zip(xy1segs, xy2segs):
        if len(xy1seg) and len(xy2seg):

            # Empty list for polygon points
            polygon_points = [] 

            # append all xy points for curve 1
            for xyvalue in xy1seg:
                polygon_points.append([xyvalue[0], xyvalue[1]]) 

            # append all xy points for curve 2 in the reverse order
            for xyvalue in xy2seg[::-1]:
                polygon_points.append([xyvalue[0], xyvalue[1]]) 

            # append the first point in curve 1 again, to it "closes" the polygon
            for xyvalue in xy1seg[0:1]:
                polygon_points.append([xyvalue[0], xyvalue[1]]) 

            p = Polygon(polygon_points).buffer(0)
            if p.geom_type == 'MultiPolygon':
                continue
            polygons.append(p)

    return MultiPolygon(polygons)


def getDirection(xy, n):
    """
    Calculates UNIT directions for each river coordinate
    This creates two columns:
        - one for the vector direction in LON
        - one for vector direction in LAT
    This is simple case that uses a forward difference model

    Inputs -
    xy (numpy array): Size is n x 2 with the centerline coordinates
    n (int): smoothing to use
    """

    cross_dlon = []
    cross_dlat = []
    tree = spatial.KDTree(xy)
    for idx, row in enumerate(xy):
        distance, neighbors = tree.query(
            [(row[0], row[1])],
            n
        )
        max_distance = np.argmax(distance[0])
        max_neighbor = neighbors[0][max_distance]
        min_distance = np.argmin(distance[0])
        min_neighbor = neighbors[0][min_distance]

        # Calculate lat and lon distances between coordinates
        distance = [
            (
                xy[max_neighbor][0]
                - xy[min_neighbor][0]
            ),
            (
                xy[max_neighbor][1]
                - xy[min_neighbor][1]
            )
        ]

        # Converts distance to unit distance
        norm = math.sqrt(distance[0] ** 2 + distance[1] ** 2)
        dlon_t, dlat_t = distance[0] / norm, distance[1] / norm
        cross_dlat.append(-1 * dlon_t)
        cross_dlon.append(dlat_t)

    return np.vstack([cross_dlon, cross_dlat]).transpose() 


def getMigrationWindow(location, direction, crosslen, crosswidth):
    # Create box to find migrated area within
    cross_offset = direction * crosslen

    down_direction = np.array([
        direction[1],
        -1* direction[0]
    ])
    down_offset = down_direction * crosswidth 

    window_points = np.empty([4, 2])
    # 1 corner
    window_points[0, 0] = (
        location[0] + down_offset[0] - cross_offset[0]
    )
    window_points[0, 1] = (
        location[1] + down_offset[1] - cross_offset[1]
    )

    # 3 corner
    window_points[1, 0] = (
        location[0] - down_offset[0] - cross_offset[0]
    )
    window_points[1, 1] = (
        location[1] - down_offset[1] - cross_offset[1]
    )

    # 4 corner
    window_points[2, 0] = (
        location[0] - down_offset[0] + cross_offset[0]
    )
    window_points[2, 1] = (
        location[1] - down_offset[1] + cross_offset[1]
    )

    # 2 corner
    window_points[3, 0] = (
        location[0] + down_offset[0] + cross_offset[0]
    )
    window_points[3, 1] = (
        location[1] + down_offset[1] + cross_offset[1]
    )

    return Polygon(window_points)


def coordToIndex(xy, xy1, xy2):
    """
    xy are the absolute coordinates of interesections
    xy1 are the cl coordinates at t1
    xy2 are the cl cooridnates at t2

    returns
    i1 intersection indeces at t1
    i2 intersection indees at t2
    """
    # T1 & T2
    i1 = []
    i2 = []
    tree1 = spatial.KDTree(xy1)
    tree2 = spatial.KDTree(xy2)
    for pair in xy:
        # T1
        distance1, n1 = tree1.query(
            pair,
            1
        )
        # T2
        distance2, n2 = tree2.query(
            pair,
            1
        )

        i1.append(n1)
        i2.append(n2)

    return i1, i2


def pickCutoffs(cl_year1, cl_year2):
    """
    Mannually pick cutoff points.
    Algorithm will find the points between the two
    """
    print('Pick Cutoffs')
    # Pick the points
    fig, ax = plt.subplots(1, 1)

    line = ax.scatter(
        cl_year1.xy[:, 0],
        cl_year1.xy[:, 1],
        color='blue',
        s=40
    )

    ax.scatter(
        cl_year2.xy[:, 0],
        cl_year2.xy[:, 1],
        color='red'
    )

    BC = PointPicker(ax)
    fig.canvas.mpl_connect('pick_event', BC)
    line.set_picker(1)

    axclear = plt.axes([0.81, 0.17, 0.1, 0.055])
    bclear = Button(axclear, 'Clear')
    bclear.on_clicked(BC.clear)

    axnext = plt.axes([0.81, 0.1, 0.1, 0.055])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(BC.next)

    axdone = plt.axes([0.81, 0.03, 0.1, 0.055])
    bdone = Button(axdone, 'Done')
    bdone.on_clicked(BC.done)
    plt.show()

    path = np.array([])
    tree = spatial.KDTree(cl_year1.xy)
    for cutoff in BC.cutoffsI:
    # Find path between the two points
        distance1, neighbor1 = tree.query(cutoff[0])
        distance2, neighbor2 = tree.query(cutoff[1])
        neighbors = [neighbor1, neighbor2]

        path = np.concatenate((path, np.array(
            nx.shortest_path(
                cl_year1.graph,
                source=neighbors[0],
                target=neighbors[1], 
                weight='length'
            )
        )))

    return path.astype(int)


def closest(lst, K):
    """
    Finds the closest value in list
    """
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


# MIGRATION
if __name__=='__main__':

    root = '/Users/greenberg/Documents/PHD/Projects/Chapter2/Rivers/Rio_Huallaga/Rio_Huallaga'
    year1 = 2011
    year2 = 2018 
    name = 'Rio_Huallaga_{}_centerline.pkl'
    out_name = f'Rio_Huallaga_{year1}_{year2}_migration.csv'

    with open(os.path.join(root, 'centerline', name.format(year1)), 'rb') as f:
        cl_year1 = pickle.load(f)

    with open(os.path.join(root, 'centerline', name.format(year2)), 'rb') as f:
        cl_year2 = pickle.load(f)

    # Load centerlines
    xy1 = cl_year1.xy
    xy2 = cl_year2.xy

    x1 = xy1[:, 0]
    y1 = xy1[:, 1]
    x2 = xy2[:, 0]
    y2 = xy2[:, 1]

    # xy1 = numpy.vstack([x1, y1]).transpose()
    # xy2 = numpy.vstack([x2, y2]).transpose()

    ix, iy = intersection(x1, y1, x2, y2)
    ixy = np.vstack([ix, iy]).transpose()
    i1, i2 = coordToIndex(ixy, xy1, xy2)
    I = np.vstack([i1, i2]).transpose()
    polygon = findMigratedArea(xy1, xy2, I)

    # for poly in polygon:
    #     plt.plot(
    #         poly.exterior.xy[0],
    #         poly.exterior.xy[1]
    #     )
    # plt.show()

    cross_dirs = getDirection(xy1, 5)
    crosslen = 300
    crosswidth = 50
    
    migration_distances = []
    for idx, (location, direction) in  enumerate(zip(xy1, cross_dirs)):
        window_poly = getMigrationWindow(
            location, 
            direction, 
            crosslen, 
            crosswidth
        )

        try:
            migrated_poly = polygon.intersection(window_poly)

            migration_distances.append(
                migrated_poly.area / (2 * crosswidth)
            )

        except errors.TopologicalError:
            migration_distances.append(None)

    # Make migration csv
    ml = np.append(
        xy1, 
        np.array(migration_distances).reshape(len(migration_distances), 1), 
        1
    )
    ml = pandas.DataFrame(
        ml, 
        columns=['Easting', 'Northing', 'Mr [m/yr]']
    )
    ml['Span [yr]'] =  int(year2) - int(year1)
    ml['Mr [m/yr]'] = ml['Mr [m/yr]'] / ml['Span [yr]'] 

    # Set cutoff points to 0
    cutoff_idxs = pickCutoffs(cl_year1, cl_year2)
    ml.at[cutoff_idxs, 'Mr [m/yr]'] = None

    # get lat-long
    in_crs = Proj(cl_year1.crs)
    out_crs = Proj(init='epsg:4326')
    ml['Lon'], ml['Lat'] = transform(
        in_crs, out_crs, ml['Easting'], ml['Northing']
    )
    ml['width'] = cl_year1.width

    ml.to_csv(
        os.path.join(root, 'migration', out_name)
    )
