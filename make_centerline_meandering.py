import os
import timeit
import math
import copy
import itertools
import pickle

import pandas
import scipy
from scipy import spatial, ndimage
from scipy import ndimage as nd
from skimage import measure, draw, morphology, feature, graph
from skimage.morphology import medial_axis, skeletonize, thin, binary_closing
from shapely.geometry import Polygon
import pyproj
from shapely.geometry import Point
from shapely.ops import transform
import numpy as np
import networkx as nx
import fiona
import rasterio
import rasterio.mask
from rasterio.plot import show
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle


class pickData(object):
    text_template = 'x: %0.2f\ny: %0.2f'
    x, y = 0.0, 0.0
    xoffset, yoffset = -20, 20
    text_template = 'x: %0.2f\ny: %0.2f'

    def __init__(self, ax, centerline_mask):
        self.ax = ax
        self.events = []
        self.points = []
        self.centerline_mask = centerline_mask 

    def clear(self, event):
        # Clear the most recent box of pointd
        self.events = []
        self.X0 = None

        # Remove all plotted picked points
        self.rect.remove()
        for p in self.points:
            if p:
                p.remove()

        # Remove most recent rectangle
        self.points = []
        self.rects = None
        print('Cleared')

    def done(self, event):
        # All done picking points
        plt.close('all')
        print('All Done')

    def draw_box(self, event):
        # Draw the points box onto the image
        width = self.events[1][0] - self.events[0][0]
        height = self.events[1][1] - self.events[0][1]
        r = Rectangle(
            self.events[0],
            width,
            height,
            color='red',
            fill=False
        )
        self.rect = self.ax.add_patch(r)

        for p in self.points:
            p.remove()
        self.points = []

        event.canvas.draw()

    def remove(self, event):
        botleft = self.rect.get_xy()
        botleft = [math.ceil(i) for i in botleft]

        # Get indexes at top right
        topright = [
            botleft[0] + self.rect.get_width(),
            botleft[1] + self.rect.get_height(),
        ]
        topright = [math.ceil(i) for i in topright]

        ys = [botleft[0], topright[0]]
        xs = [botleft[1], topright[1]]

        self.centerline_mask[
            min(xs):max(xs),
            min(ys):max(ys)
        ] = False

        self.ax.imshow(self.centerline_mask)
        event.canvas.draw()

    def __call__(self, event):
        # Call the events
        self.event = event

        if not event.dblclick:
            return 0

        self.x, self.y = event.xdata, event.ydata 
        self.events.append((self.x, self.y))
        self.events = list(set(self.events))

        if self.x is not None:
            # Plot where the picked point is
            self.points.append(self.ax.scatter(self.x, self.y))
            event.canvas.draw()

        if len(self.events) == 2:
            self.draw_box(event)
            self.events = []


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

    def find_intersections(self):
        rr, cc = np.where(self.mask)

        rows = []
        cols = []
        # Get neighboring pixels
        for r, c in zip(rr, cc):
            window = self.mask[r-1:r+2, c-1:c+2]

            # if len(window[window]) > 3:
            if np.sum(window) > 3:
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

            # if len(window[window]) < 3:
            if np.sum(window) < 3:
                rows.append(r)
                cols.append(c)

        return np.array([cols, rows]).transpose()


    def remove_small_segments(self, intersections, endpoints, thresh):
        tree = spatial.KDTree(intersections)
        costs = np.where(self.mask, 1, 1)
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

    def filter_centerline(self, thresh=5):

        labels = measure.label(centerline.mask)
        bins = np.bincount(labels.flat)[1:] 
        filt = np.argwhere(bins >= thresh) + 1
        for f in filt:
            labels[np.where(labels == f)] = 9999
        labels[labels != 9999] = 0
        labels[labels == 9999] = 1

        self.mask = labels

    def prune_centerline(self, es, thresh=10):
        removed = 999
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
        self.idx = np.array(np.where(self.mask)).T

    def get_xy(self):
        self.xy = np.array(rasterio.transform.xy(
            self.transform,
            self.idx[:, 0], 
            self.idx[:, 1], 
        )).T

    def get_graph(self):

        start = 0
        end = len(self.idx)-1
        tmp = [tuple(i) for i in self.idx]

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

    def graph_sort(self, es):

        if es == 'EW':
            start = np.where(self.xy[:, 0] == self.xy[:, 0].min())[0][0]
            end = np.where(self.xy[:, 0] == self.xy[:, 0].max())[0][0]

        if es == 'NS':
            start = np.where(self.xy[:, 1] == self.xy[:, 1].min())[0][0]
            end = np.where(self.xy[:, 1] == self.xy[:, 1].max())[0][0]

        # Sort the shuffled DataFrame
        path = np.array(
            nx.shortest_path(self.graph, source=start, target=end, weight='length')
        )

        self.xy = self.xy[path]
        self.graph = self.graph.subgraph(path)
        mapping = dict(zip(path, range(0, len(path))))
        self.graph = nx.relabel_nodes(self.graph, mapping)

    def smooth_coordinates(self, window=5, poly=1):
        smoothed = savgol_filter(
            (self.xy[:, 0], self.xy[:, 1]), 
            window, 
            poly
        ).transpose()

        return np.vstack([smoothed[:, 0], smoothed[:, 1]]).T

    def manually_clean(self):
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(self.mask)
        t = plt.gca()
        PD = pickData(t, self.mask)
    
        axclear = plt.axes([0.0, 0.0, 0.1, 0.1])
        bclear = Button(plt.gca(), 'Clear')
        bclear.on_clicked(PD.clear)

        axremove = plt.axes([0.1, 0.0, 0.1, 0.1])
        bremove = Button(plt.gca(), 'Remove')
        bremove.on_clicked(PD.remove)

        axdone = plt.axes([0.2, 0.0, 0.1, 0.1])
        bdone = Button(plt.gca(), 'Done')
        bdone.on_clicked(PD.done)

        fig.canvas.mpl_connect('button_press_event', PD)

        im.set_picker(5) # Tolerance in points

        plt.show()

        self.centerline_clean = PD.centerline_mask


def fill_holes(mask, thresh=40):

    # Find contours
    contours = measure.find_contours(mask, 0.5)
    # Display the image and plot all contours found
    polys = []
    for contour in contours:
        # Get polygon
        poly = Polygon(contour)
        area = poly.area 

        # Filter by size
        if area <= thresh:
            polys.append(poly)

    holes = np.zeros(mask.shape)
    holesi = np.where(rasterio.features.rasterize(
        polys, out_shape=mask.shape, all_touched=True
    ))
    holes[holesi[1], holesi[0]] = 1
    holes = nd.binary_erosion(holes)

    mask = np.copy(holes + mask)
    mask[mask > 0] = 1
    mask = nd.binary_closing(mask)

    return mask


def get_centerline(mask, smoothing):

    labels = measure.label(mask)
     # assume at least 1 CC
    assert( labels.max() != 0 )

    # Find largest connected component
    bins = np.bincount(labels.flat)[1:] 
    cc = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    filt = ndimage.maximum_filter(cc, size=smoothing)
    # Find skeletonized centerline
    # skeleton = skeletonize(filt, method='lee')
    skeleton = skeletonize(mask, method='lee')
    skeleton = thin(skeleton)

    return skeleton 


def JoinComponents(H, G):
    G.remove_edges_from(list(G.edges))
    # Get number of components
    S = [H.subgraph(c).copy() for c in nx.connected_components(H)]
    while len(S) != 1:
        # Get node positions
        node_attributes = nx.get_node_attributes(H, 'pos')

        # Iterate through all pairs ofconnected components
        Scombs = list(itertools.product(S, S))
        for idx, pair in enumerate(Scombs):
            if pair[0] == pair[1]:
                continue
            # Get all combinations between two lists
            S0_nodes = list(pair[0].nodes)
            S1_nodes = list(pair[1].nodes)
            combs = list(itertools.product(S0_nodes, S1_nodes))

            # Get all lengths between each graph nodes
            lengths = []
            for comb in combs:
                pos1 = node_attributes[comb[0]]
                pos2 = node_attributes[comb[1]]
                lengths.append(np.linalg.norm(np.array(pos1) - np.array(pos2)))

            # Get shortest length index and use that edge
            i = np.argsort(lengths)[0]
            comp_edge = combs[i]
            length = lengths[i]

            G.add_edge(*comp_edge, length=length)

        # Iterate trhough components to find shortest component for each node
        for s in S:
            comp_nodes = list(s.nodes)
            min_edges = []
            min_lengths = []
            for node in comp_nodes:
                edges = list(G.edges(node))
                lengths = [G.get_edge_data(*e)['length'] for e in edges]
                if not lengths:
                    continue
                min_edges.append(edges[np.argmin(lengths)])
                min_lengths.append(lengths[np.argmin(lengths)])

            H.add_edge(
                *min_edges[np.argmin(min_lengths)], 
                length=np.min(min_lengths)
            )

        G.remove_edges_from(list(G.edges))
        # Get number of components
        S = [H.subgraph(c).copy() for c in nx.connected_components(H)]
        print(len(S))

    return H


def create_channel_polygon(contours):
    polygons = []
    longest = 0
    # Make polygons and find the longest contour
    for i, contour in enumerate(contours):
        polygons.append(Polygon(contour))

    # find which polygons fall within the longest
    inners = []
    for i, polygon in enumerate(polygons):
        # Save the river polygon
        river_polygon = polygons[longest_i]

        # If the polygon is the river polygon move to the next
        # if i == longest_i:
        #     continue

        # See if the polygon is within the river polygons
        if river_polygon.contains(polygon):
            inners.append(polygon)

    return Polygon(
        river_polygon.exterior.coords, 
        [inner.exterior.coords for inner in inners]
    )


def calculate_width(centerline, image):
    lengths = nx.shortest_path_length(
        centerline.graph, 
        weight='length'
    )
    pairs = []
    for start, dists in lengths:
        # start, dists = length
        max_dist = np.max(list(dists.values()), axis=0)
        end = list(
            dists.keys()
        )[np.where(list(dists.values()) == max_dist)[0][0]]
        pairs.append((start, end, max_dist))

    pairs = np.array(pairs)
    start, end, dist = pairs[np.argmax(pairs[:,2]), :]
    area = len(np.where(image)[0]) * 30 * 30
    # Width in meters
    return area / (dist * 30)


def get_largest(mask):
    labels = measure.label(mask)
     # assume at least 1 CC
    assert(labels.max() != 0)

    # Find largest connected component
    bins = np.bincount(labels.flat)[1:] 

    return labels == np.argmax(np.bincount(labels.flat)[1:])+1


def get_widths(centerline, image, scale=5):
    def get_direction(pos, scale=5):

        dy = (pos[1,0] - pos[0,0]) * scale
        dx = (pos[1,1] - pos[0,1]) * scale

        return dx, dy
    
    def get_cross_section_area(dx, dy, pos):
        # top_left = (pos[0, 0] + dx, pos[0, 1] - dy)
        # top_right = (pos[0, 0] - dx, pos[0, 1] + dy)
        # bot_left = (pos[1, 0] + dx, pos[1, 1] - dy)
        # bot_right = (pos[1 ,0] - dx, pos[1, 1] + dy)

        top_left = (pos[0, 1] - dy, pos[0, 0] + dx)
        top_right = (pos[0, 1] + dy, pos[0, 0] - dx)
        bot_left = (pos[1, 1] - dy, pos[1, 0] + dx)
        bot_right = (pos[1, 1] + dy, pos[1 ,0] - dx)

        corners = np.array([
            top_left,
            top_right,
            bot_right,
            bot_left,
        ])

        return Polygon(corners)
    
    def get_largest(image):
        labels = measure.label(image)
         # assume at least 1 CC
        assert( labels.max() != 0 )
        # Find largest connected component
        bins = np.bincount(labels.flat)[1:] 
        channel = labels == np.argmax(np.bincount(labels.flat)[1:])+1

        return channel

    allpos = nx.get_node_attributes(centerline.graph, 'pos')
    widths = np.empty([len(centerline.graph.nodes), 6])
    for node in centerline.graph.nodes:
        print(node)
        n1 = [i for i in centerline.graph.neighbors(node)]
        n = []
        for n_i in n1:
            n += [i for i in centerline.graph.neighbors(n_i)]

        if len(n) == 1:
            pos = [allpos[node]]
        elif len(n) > 1:
            pos = []
        for i in n:
            pos.append(allpos[i])
        pos = np.array(pos)
        length = np.linalg.norm(pos[0,:]-pos[-1,:])
        pos = np.array([pos[0,:], pos[-1,:]])

        dx, dy = get_direction(pos, scale=scale)
        poly = get_cross_section_area(dx, dy, pos)
        if not poly.area:
            continue
        area = rasterio.features.rasterize(
            [poly], 
            out_shape=image.shape
        )
        areai = np.where(area)
        crop = np.copy(image) + np.copy(area)
        crop[~(crop == 2)] = 0

        area = get_largest(crop)

        # if node % 1000 == 0:
        #     plt.imshow(image + area)
        #     plt.scatter(pos[:,1], pos[:,0])
        #     plt.show()

        width = (np.sum(area)) / length

        widths[node,:] = [
            node, 
            allpos[node][0],
            allpos[node][1],
            centerline.xy[node,0],
            centerline.xy[node,1],
            width
        ]

    return widths


if __name__=='__main__':

    # root = '/home/greenberg/ExtraSpace/PhD/Projects/ComparativeMobility/Rivers/TorsaDownstream/ML_masks/Torsa1'
    root = '/Users/greenberg/Documents/PHD/Projects/Chapter2/Rivers/Indus/Indus/mask'
    # name = f'Torsa1_{year}_mask.tif'
    name = f'Indus_1990_01-01_12-31_mask.tif'
    ipath = os.path.join(root, name)
    ds = rasterio.open(ipath)
    
    mask = ds.read(1)
    # mask = get_largest(mask)
    image = fill_holes(np.copy(mask), 10000).astype(int)
    
    centerline = Centerline(
        get_centerline(image, 5), 
        ds.crs, 
        ds.transform
    )
    centerline.filter_centerline(5)
    # centerline.prune_centerline('NS', 5)
    # centerline.manually_clean()
    centerline.get_idx()
    centerline.get_xy()
    centerline.get_graph()
    centerline.graph_sort('NS')
    
    widths = get_widths(centerline, image, scale=4)
    width = calculate_width(centerline, image)
    centerline.width = width
    centerline.point_width = widths
    
    # out_root = '/home/greenberg/ExtraSpace/PhD/Projects/ComparativeMobility/Rivers/TorsaDownstream/centerlines/Torsa1'
    out_root = '/Users/greenberg/Code/Github/MakeCenterline/example'
    out_name = f'Chindwin_2019_centerline.pkl'
    # out_name = f'Torsa1_{year}_centerline.pkl'
    out_path = os.path.join(out_root, out_name)
    with open(out_path, 'wb') as f:
        pickle.dump(centerline, f)
