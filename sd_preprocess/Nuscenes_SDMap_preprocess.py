import os 
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from mmcv import Config

from nuscenes import NuScenes
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString

import torch
from torch.utils.data import Dataset

from Nuscenes_SDMap_utils import CLASS2LABEL, get_scenes, get_samples

class VectorizedLocalMap(object):
    def __init__(self,
                 dataroot,
                 patch_size,
                 canvas_size,
                 sd_map_path='./nuscenes_osm',
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 normalize=False,
                 fixed_num=-1):
        '''
        Args:
            fixed_num = -1 : no fixed num
        '''
        super().__init__()
        self.data_root = dataroot
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.line_classes = line_classes 
        self.ped_crossing_classes = ped_crossing_classes 
        self.polygon_classes = contour_classes 
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.normalize = normalize
        self.fixed_num = fixed_num
        self.sd_maps = {}
        proj = 3857
        options = ['trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', # road
                'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'# road link
                'living_street',  'road',  # Special road  'service'
                ]
        
        map_origin_df = pd.DataFrame(
            {'City': ['boston-seaport', 'singapore-onenorth', 'singapore-hollandvillage', 'singapore-queenstown'],
            'Latitude': [42.336849169438615, 1.2882100868743724, 1.2993652317780957, 1.2782562240223188],
            'Longitude': [-71.05785369873047, 103.78475189208984, 103.78217697143555, 103.76741409301758]}) #
        map_origin_gdf = gpd.GeoDataFrame(
            map_origin_df, geometry=gpd.points_from_xy(map_origin_df.Longitude, map_origin_df.Latitude), crs=4326)
        map_origin_gdf = map_origin_gdf.to_crs(proj) 
        for loc in self.MAPS:
            sd_map = gpd.read_file(os.path.join(sd_map_path, '{}.shp'.format(loc)))
            sd_map = sd_map.to_crs(proj)
            sd_map = sd_map[sd_map['type'].isin(options)]
            sd_map = MultiLineString(list(sd_map.geometry))
            origin_geo = map_origin_gdf[map_origin_gdf['City']==loc].geometry

            # before futureWarning : origin = (float(origin_geo.x), float(origin_geo.y))
            origin = (float(origin_geo.x.iloc[0]), float(origin_geo.y.iloc[0]))

            matrix = [1.0, 0.0, 0.0, 1.0, -origin[0], -origin[1]]
            self.sd_maps[loc] = affinity.affine_transform(sd_map, matrix)
            if loc == 'boston-seaport':
                scale = 0.7143
                matrix = [scale, 0.0, 0.0, scale, 0.0, 0.0]
                self.sd_maps[loc] = affinity.affine_transform(self.sd_maps[loc], matrix)
                matrix = [1.0351, 0.0014, -0.0002, 1.0326, 0.0, 0.0]
                self.sd_maps[loc] = affinity.affine_transform(self.sd_maps[loc], matrix)
                
    def get_osm_geom(self, patch_box, patch_angle, location):
        osm_map = self.sd_maps[location]
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        for geom_line in osm_map.geoms:
            if geom_line.is_empty:
                continue
            new_line = geom_line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)
        return line_list

    def gen_vectorized_samples(self, location, ego2global_translation, ego2global_rotation):
        map_pose = ego2global_translation[:2]
        rotation = Quaternion(ego2global_rotation)
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location) 
        line_vector_dict = self.line_geoms_to_vectors(line_geom)
        ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
        ped_vector_list = self.line_geoms_to_vectors(ped_geom)['ped_crossing']
        polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
        poly_bound_list, union_roads = self.poly_geoms_to_vectors(polygon_geom)

        vectors = []
        for line_type, vects in line_vector_dict.items():
            for line, length in vects:
                vectors.append((line.astype(float), length, CLASS2LABEL.get(line_type, -1))) 
        for ped_line, length in ped_vector_list:
            vectors.append((ped_line.astype(float), length, CLASS2LABEL.get('ped_crossing', -1)))

        for contour, length in poly_bound_list:
         
            vectors.append((contour.astype(float), length, CLASS2LABEL.get('contours', -1)))

        filtered_vectors = []
        for pts, pts_num, type in vectors:
            if type != -1:
                filtered_vectors.append({
                    'pts': pts,
                    'pts_num': pts_num,
                    'type': type
                })

        osm_geom = self.get_osm_geom(patch_box, patch_angle, location)
        osm_vector_list = []

        def is_intersection(line):
            if line.intersects(union_roads):
                return True
            else:
                return False
            
        for line in osm_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        osm_vector_list.append(self.sample_fixed_pts_from_line(single_line, padding=False, fixed_num=50))
                elif line.geom_type == 'LineString':
                    osm_vector_list.append(self.sample_fixed_pts_from_line(line, padding=False, fixed_num=50))
                else:
                    raise NotImplementedError

        return filtered_vectors, polygon_geom, osm_vector_list 

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.map_explorer[location]._get_layer_line(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, patch_angle, location)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(single_line))
                elif line.geom_type == 'LineString':
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1] 
        lanes = polygon_geom[1][1] 
        union_roads = ops.unary_union(roads) 
        union_lanes = ops.unary_union(lanes) 
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior) 
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw: 
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch) 
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results), union_roads

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != 'MultiPolygon':
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list):
            points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx:idx + 2], poly_xy[1, idx:idx + 2])]
            line = LineString(points)
            line = line.intersection(patch)
            if not line.is_empty:
                line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(line)

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].extract_polygon(record['polygon_token'])
            poly_xy = np.array(polygon.exterior.xy)
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]

            add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y, line_list)
            add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y, line_list)

        return line_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        else:
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

        if self.normalize:
            sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            return sampled_points, num_valid
        num_valid = len(sampled_points)
        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[:self.num_samples, :]
                num_valid = self.num_samples

            if self.normalize:
                sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
                num_valid = len(sampled_points)

        return sampled_points, num_valid

    def sample_fixed_pts_from_line(self, line, padding=False, fixed_num=100):
        if padding:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
        else:
            distances = np.linspace(0, line.length, fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)

        num_valid = len(sampled_points)

        if num_valid < fixed_num:
            padding = np.zeros((fixed_num - len(sampled_points), 2))
            sampled_points = np.concatenate([sampled_points, padding], axis=0)
        elif num_valid > fixed_num:
            sampled_points = sampled_points[:fixed_num, :]
            num_valid = fixed_num
        if self.normalize:
            sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
            num_valid = len(sampled_points)

        num_valid = len(sampled_points)
        return sampled_points, num_valid

class SDMap_VectorExtraction(Dataset):
    def __init__(self, version, dataroot, data_conf, is_train):
        super(SDMap_VectorExtraction, self).__init__()
        patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]  
        patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]  
        canvas_h = int(patch_h / data_conf['ybound'][2])           
        canvas_w = int(patch_w / data_conf['xbound'][2]) 

        self.is_train = is_train
        self.data_conf = data_conf
        self.patch_size = (patch_h, patch_w)    
        self.canvas_size = (canvas_h, canvas_w)
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.vector_map = VectorizedLocalMap(dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size, 
                                             sd_map_path=data_conf['sd_map_path'])
        self.scenes = get_scenes(version=version, is_train=self.is_train)
        self.samples = get_samples(self.nusc, self.scenes)
    
    def __len__(self):
        return len(self.samples)

    def get_vectors(self, rec):
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location'] 
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token']) 
        vectors, polygon_geom, osm_vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        return vectors, polygon_geom, osm_vectors

    def get_sd_map(self, rec):
        vectors, polygon_geom, osm_vectors = self.get_vectors(rec) 
        return vectors, polygon_geom, osm_vectors

    def __getitem__(self, idx):
        rec = self.samples[idx] 
        timestamp = torch.tensor(rec['timestamp'])
        scene_token = rec['scene_token']
        scene_id = self.nusc.get('scene', scene_token)['name']
    
        vector, polygon_geom, osm_vectors = self.get_sd_map(rec)
        return vector, polygon_geom, osm_vectors, timestamp, scene_id

def pdist(A, squared = False, eps = 1e-8):
    prod = torch.mm(A,A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2*prod)
    if squared:
        return res
    else:
        res = res.sqrt()
        return res

def preprocess(data_conf,
               is_train,
               version,
               path_to_sdmap = '../data/nuscenes/osm',
               dataset_root = '../data/nuscenes'):

    dataroot = os.path.join(dataset_root)
    dataset = SDMap_VectorExtraction(version=version, 
                                     dataroot=dataroot, 
                                     data_conf=data_conf, 
                                     is_train=is_train)
    for idx in tqdm(range(dataset.__len__())):
        vector, polygon_geom, osm_vectors, timestamp, scene_id = dataset.__getitem__(idx)

        xs = []
        ys = []
        if osm_vectors != []:
            for each in osm_vectors:
                # each[0] contains centerline points 
                xs.extend([each[0][0,0],each[0][-1,0]])
                ys.extend([each[0][0,1],each[0][-1,1]])
            lane_pts = torch.tensor([xs,ys]).t()
            adj = pdist(lane_pts)
            edge = (~(adj > 1e-4)).float()
            edge = edge - torch.eye(edge.shape[0])
            edge_tmp = torch.stack([edge[2*i,:]+edge[2*i+1,:] for i in range(len(xs)//2)],dim=0)
            edge_final = torch.stack([edge_tmp[:,2*i]+edge_tmp[:,2*i+1] for i in range(len(ys)//2)],dim=1)
        else:
            edge_final = None

        sd_map = {'vector':vector,
                  'polygon_geom': polygon_geom,
                  'osm_vectors':osm_vectors,
                  'timestamp':timestamp,
                  'scene_id':scene_id,
                  'edge':edge_final}
              
        sd_map_root_path = dataroot
        dump_file_root = (os.path.join(sd_map_root_path, 'sd_map'))
        if not(os.path.isdir(dump_file_root)):
            os.mkdir(dump_file_root)

        dump_file_ = '{}/{}.pkl'.format(dump_file_root,str(timestamp.item()))
        if os.path.isfile(dump_file_):
            continue
        save_file=open(dump_file_,'wb')
        pickle.dump(sd_map,save_file)
        break

def preprocess_2(data_conf,
               is_train,
               version,
               path_to_sdmap = '../data/nuscenes/osm',
               dataset_root = '../data/nuscenes'):
    dataroot = os.path.join(dataset_root)
    dataset = SDMap_VectorExtraction(version=version, 
                                     dataroot=dataroot, 
                                     data_conf=data_conf, 
                                     is_train=is_train)
    for idx in tqdm(range(dataset.__len__())):
        sd_map_root_path = dataroot
        vector, polygon_geom, osm_vectors, timestamp, scene_id = dataset.__getitem__(idx)
        dump_file_root = (os.path.join(sd_map_root_path, 'sd_map'))
        dump_file_ = '{}/{}.pkl'.format(dump_file_root,str(timestamp.item()))
        save_file=open(dump_file_,'wb')

  
if __name__ == '__main__':
    path_to_sdmap = './data/nuscenes/osm'
    data_conf = {
            'image_size': (900, 1600),
            'xbound': [-30.0, 30.0, 0.15],
            'ybound': [-15.0, 15.0, 0.15],
            'thickness': 5,
            'angle_class': 36,
            'sd_map_path':path_to_sdmap
        }
    
    is_train_list = [False,True]
    is_train_state = {True:'trainval',False:'test'}

    # 设定读取数据集所属的version, sdmap只在 v1.0-mini 中存在
    # version : 'v1.0-trainval', 'v1.0-mini', 'v1.0-test'
    version_list = ['v1.0-mini', 'v1.0-trainval', 'v1.0-test']

    # the path to dataset containing [v1.0-mini, train, val, test], each split contains [lidar, imgs, sdmap]
    dataset_root = './data/nuscenes'
    for version in version_list:
        for is_train in is_train_list:
            """try:
                print('{}.{} is precessing'.format(version,is_train_state[is_train]))
                preprocess(data_conf,is_train,version,dataset_root=dataset_root)
            except:
                print('Cannot find {}.{} file'.format(version,is_train_state[is_train]))
                continue"""
            preprocess(data_conf,is_train,version,dataset_root=dataset_root)