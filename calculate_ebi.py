from scipy import spatial
from scipy import interpolate
import pickle
import numpy as np
import pandas
from matplotlib import pyplot as plt
import os
import rasterio
from sklearn.linear_model import LinearRegression


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


root = '/home/greenberg/ExtraSpace/PhD/Projects/ComparativeMobility/Rivers/TorsaDownstream/centerlines/Torsa1'
name = 'Torsa1_2021_centerline.pkl'
path = os.path.join(root, name)

with open(path, 'rb') as f:
    centerline = pickle.load(f)

widths = pandas.DataFrame(
    centerline.point_width, 
    columns=['Node', 'Row', 'Col', 'Easting', 'Northing', 'Width [px]']
)
widths = widths[widths['Row'] > 1]

tree = spatial.KDTree(widths[['Row', 'Col']])
window = 30
es = 'NS'
for i, row in widths.iterrows():
    dist, n = tree.query(row[['Row', 'Col']], window)
    sample = widths.iloc[n]
    x = sample['Col'].values.reshape(-1,1)
    y = np.expand_dims(sample['Row'].values, 1)
    if es == 'NS':
        mod = LinearRegression().fit(y, x)
    else:
        mod = LinearRegression().fit(x, y)
    lin = lambda x: (x * mod.coef_[0][0]) + mod.intercept_[0]
    inv_lin = lambda x, y: (
        (-1 * x * (1 / mod.coef_[0][0])) 
        + y
    )
    break

spl = interpolate.UnivariateSpline(widths['Row'], widths['Col'])

plt.imshow(centerline.mask)
plt.scatter(widths['Col'], widths['Row'])
plt.scatter(sample['Col'], sample['Row'])
if es == 'NS':
    plt.plot(lin(sample['Row'].values), sample['Row'].values)
else:
    plt.plot(sample['Col'].values, lin(sample['Col'].values))

### 
# NEED TO FIX THE ORIENTATION OF THE RIVER IN REGARDS TO HOW THE CB LINE IS DRAWN.  
# NEED TO LINE UP EACH CROSS SECTION WITH THE CORRECT PIXEL SAMPLING
# NEED TO RE-RUN ALL TORSA CENTERLINE PULLS
length = 10 
for row, col in zip(sample['Row'].values, sample['Col'].values):
    if es == 'NS':
        x = np.arange(row - length, row + length)
        y = np.array([col for i in x])
        plt.plot(
            inv_lin(x, y),
            x
        )
    else:
        x = np.arange(col - length, col + length)
        y = np.array([row for i in x])
        plt.plot(
            x, 
            inv_lin(x, y)
        )
plt.show()
