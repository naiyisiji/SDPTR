import copy

import numpy as np
from mmdet.datasets import DATASETS
import mmcv
import os
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmcv.parallel import DataContainer as DC
import random

from .nuscenes_dataset import CustomNuScenesDataset
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from shapely.geometry import LineString
from mmdet.datasets.pipelines import to_tensor
from pandas import read_pickle
import json

from .nuscenes_map_dataset import (VectorizedLocalMap,LiDARInstanceLines,
                                  add_rotation_noise,add_translation_noise,
                                  output_to_vecs,sample_pts_from_line)

@DATASETS.register_module()
class NuScenesSDmapDataset(CustomNuScenesDataset):
    r"""NuScenes Dataset.

    Args:
        img_lidar_datset
        sdmap_sataset

    """
    MAPCLASSES = ('divider',)
    def __init__(self,
                 img_lidar_dataset=None,
                 sdmap_dataset=None,
                 map_ann_file=None, 
                 queue_length=4, 
                 bev_size=(200, 200), 
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 overlap_test=False, 
                 fixed_ptsnum_per_line=-1,
                 eval_use_same_gt_sample_num_flag=False,
                 padding_value=-10000,
                 map_classes=None,
                 noise='None',
                 noise_std=0,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        # adding 2 dataset
        self.img_lidar_dataset = img_lidar_dataset
        self.sdmap_dataset = sdmap_dataset

        # other setting in CustomNuScenesDataset
        self.map_ann_file = map_ann_file
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4]-pc_range[1]
        patch_w = pc_range[3]-pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = eval_use_same_gt_sample_num_flag
        self.vector_map = VectorizedLocalMap(kwargs['data_root'], 
                            patch_size=self.patch_size, map_classes=self.MAPCLASSES, 
                            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
                            padding_value=self.padding_value)
        
        self.data_root = kwargs['data_root']
        self.is_vis_on_test = False
        self.noise = noise
        self.noise_std = noise_std
    @classmethod
    def get_map_classes(cls, map_classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if map_classes is None:
            return cls.MAPCLASSES

        if isinstance(map_classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(map_classes)
        elif isinstance(map_classes, (tuple, list)):
            class_names = map_classes
        else:
            raise ValueError(f'Unsupported type {type(map_classes)} of map classes.')

        return class_names
    def vectormap_pipeline(self, example, input_dict):
        '''
        `example` type: <class 'dict'>
            keys: 'img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img';
                  all keys type is 'DataContainer';
                  'img_metas' cpu_only=True, type is dict, others are false;
                  'gt_labels_3d' shape torch.size([num_samples]), stack=False,
                                padding_value=0, cpu_only=False
                  'gt_bboxes_3d': stack=False, cpu_only=True
        '''
        # import pdb;pdb.set_trace()
        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(input_dict['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = input_dict['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(input_dict['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = input_dict['ego2global_translation']

        lidar2global = ego2global @ lidar2ego

        lidar2global_translation = list(lidar2global[:3,3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

        location = input_dict['map_location']
        ego2global_translation = input_dict['ego2global_translation']
        ego2global_rotation = input_dict['ego2global_rotation']
        anns_results = self.vector_map.gen_vectorized_samples(location, lidar2global_translation, lidar2global_rotation)
        
        '''
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index
        '''
        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        else:
            gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
            except:
                # empty tensor, will be passed in train, 
                # but we preserve it for test
                gt_vecs_pts_loc = gt_vecs_pts_loc
        example['gt_labels_3d'] = DC(gt_vecs_label, cpu_only=False)
        example['gt_bboxes_3d'] = DC(gt_vecs_pts_loc, cpu_only=True)
        return example

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []

        # temporal aug
        prev_indexs_list = list(range(index-self.queue_length, index))
        random.shuffle(prev_indexs_list)
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
        ##

        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        example = self.vectormap_pipeline(example,input_dict)
        
        if self.filter_empty_gt and \
                (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None
        data_queue.insert(0, example)
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                example = self.vectormap_pipeline(example,input_dict)
                if self.filter_empty_gt and \
                        (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                    return None
                frame_idx = input_dict['frame_idx']
            data_queue.insert(0, copy.deepcopy(example))
        prepared_train_data = self.union2one(data_queue)    
        return prepared_train_data

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if i == 0:
                metas_map[i]['prev_bev'] = False
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]['img'] = DC(torch.stack(imgs_list),
                              cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            lidar_path=info["lidar_path"],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'],
            map_location = info['map_location'],
        )
        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        input_dict["lidar2ego"] = lidar2ego
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            input_dict["camera2ego"] = []
            input_dict["camera_intrinsics"] = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                lidar2cam_rt_t = lidar2cam_rt.T

                if self.noise == 'rotation':
                    lidar2cam_rt_t = add_rotation_noise(lidar2cam_rt_t, std=self.noise_std)
                elif self.noise == 'translation':
                    lidar2cam_rt_t = add_translation_noise(
                        lidar2cam_rt_t, std=self.noise_std)

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt_t)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt_t)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    cam_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = cam_info["sensor2ego_translation"]
                input_dict["camera2ego"].append(camera2ego)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = cam_info["cam_intrinsic"]
                input_dict["camera_intrinsics"].append(camera_intrinsics)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        # load lidar dataset
        if self.modality['use_lidar']:
            pts_filename = osp.join(self.data_root, info['lidar_path'])
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename
        
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle


        lidar2ego = np.eye(4)
        lidar2ego[:3,:3] = Quaternion(input_dict['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = input_dict['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3,:3] = Quaternion(input_dict['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = input_dict['ego2global_translation']
        lidar2global = ego2global @ lidar2ego
        input_dict['lidar2global'] = lidar2global

        # load sd_map
        sdmap_path = os.path.join(self.data_root,'sd_map',str(input_dict['timestamp']) + '.pkl')
        sd_map = read_pickle(sdmap_path)['osm_vectors']
        edges = read_pickle(sdmap_path)['edge']
        input_dict['sdmap'] = sd_map
        input_dict['edge'] = edges
        return input_dict

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.is_vis_on_test:
            example = self.vectormap_pipeline(example, input_dict)
        return example

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                print('Input data is None')
                idx = self._rand_another(idx)
                continue
            return data
    def _format_gt(self):
        gt_annos = []
        print('Start to convert gt map format...')
        assert self.map_ann_file is not None
        if (not os.path.exists(self.map_ann_file)) :
            dataset_length = len(self)
            prog_bar = mmcv.ProgressBar(dataset_length)
            mapped_class_names = self.MAPCLASSES
            for sample_id in range(dataset_length):
                sample_token = self.data_infos[sample_id]['token']
                gt_anno = {}
                gt_anno['sample_token'] = sample_token
                # gt_sample_annos = []
                gt_sample_dict = {}
                gt_sample_dict = self.vectormap_pipeline(gt_sample_dict, self.data_infos[sample_id])
                gt_labels = gt_sample_dict['gt_labels_3d'].data.numpy()
                gt_vecs = gt_sample_dict['gt_bboxes_3d'].data.instance_list
                gt_vec_list = []
                for i, (gt_label, gt_vec) in enumerate(zip(gt_labels, gt_vecs)):
                    name = mapped_class_names[gt_label]
                    anno = dict(
                        pts=np.array(list(gt_vec.coords)),
                        pts_num=len(list(gt_vec.coords)),
                        cls_name=name,
                        type=gt_label,
                    )
                    gt_vec_list.append(anno)
                gt_anno['vectors']=gt_vec_list
                gt_annos.append(gt_anno)

                prog_bar.update()
            nusc_submissions = {
                'GTs': gt_annos
            }
            print('\n GT anns writes to', self.map_ann_file)
            mmcv.dump(nusc_submissions, self.map_ann_file)
        else:
            print(f'{self.map_ann_file} exist, not update')

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        assert self.map_ann_file is not None
        pred_annos = []
        mapped_class_names = self.MAPCLASSES
        # import pdb;pdb.set_trace()
        print('Start to convert map detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            pred_anno = {}
            vecs = output_to_vecs(det)
            sample_token = self.data_infos[sample_id]['token']
            pred_anno['sample_token'] = sample_token
            pred_vec_list=[]
            for i, vec in enumerate(vecs):
                name = mapped_class_names[vec['label']]
                anno = dict(
                    pts=vec['pts'],
                    pts_num=len(vec['pts']),
                    cls_name=name,
                    type=vec['label'],
                    confidence_level=vec['score'])
                pred_vec_list.append(anno)

            pred_anno['vectors'] = pred_vec_list
            pred_annos.append(pred_anno)


        if not os.path.exists(self.map_ann_file):
            self._format_gt()
        else:
            print(f'{self.map_ann_file} exist, not update')

        nusc_submissions = {
            'meta': self.modality,
            'results': pred_annos,

        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'nuscmap_results.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def to_gt_vectors(self,
                      gt_dict):
        # import pdb;pdb.set_trace()
        gt_labels = gt_dict['gt_labels_3d'].data
        gt_instances = gt_dict['gt_bboxes_3d'].data.instance_list

        gt_vectors = []

        for gt_instance, gt_label in zip(gt_instances, gt_labels):
            pts, pts_num = sample_pts_from_line(gt_instance, patch_size=self.patch_size)
            gt_vectors.append({
                'pts': pts,
                'pts_num': pts_num,
                'type': int(gt_label)
            })
        vector_num_list = {}
        for i in range(self.NUM_MAPCLASSES):
            vector_num_list[i] = []
        for vec in gt_vectors:
            if vec['pts_num'] >= 2:
                vector_num_list[vec['type']].append((LineString(vec['pts'][:vec['pts_num']]), vec.get('confidence_level', 1)))
        return gt_vectors

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='chamfer',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from projects.mmdet3d_plugin.datasets.map_utils.mean_ap import eval_map
        from projects.mmdet3d_plugin.datasets.map_utils.mean_ap import format_res_gt_by_classes
        result_path = osp.abspath(result_path)
        detail = dict()
        
        print('Formating results & gts by classes')
        with open(result_path,'r') as f:
            pred_results = json.load(f)
        gen_results = pred_results['results']
        with open(self.map_ann_file,'r') as ann_f:
            gt_anns = json.load(ann_f)
        annotations = gt_anns['GTs']
        cls_gens, cls_gts = format_res_gt_by_classes(result_path,
                                                     gen_results,
                                                     annotations,
                                                     cls_names=self.MAPCLASSES,
                                                     num_pred_pts_per_instance=self.fixed_num,
                                                     eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,
                                                     pc_range=self.pc_range)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        for metric in metrics:
            print('-*'*10+f'use metric:{metric}'+'-*'*10)

            if metric == 'chamfer':
                thresholds = [0.5,1.0,1.5]
            elif metric == 'iou':
                thresholds= np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds),self.NUM_MAPCLASSES))

            for i, thr in enumerate(thresholds):
                print('-*'*10+f'threshhold:{thr}'+'-*'*10)
                mAP, cls_ap = eval_map(
                                gen_results,
                                annotations,
                                cls_gens,
                                cls_gts,
                                threshold=thr,
                                cls_names=self.MAPCLASSES,
                                logger=logger,
                                num_pred_pts_per_instance=self.fixed_num,
                                pc_range=self.pc_range,
                                metric=metric)
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']

            for i, name in enumerate(self.MAPCLASSES):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail['NuscMap_{}/{}_AP'.format(metric,name)] =  cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail['NuscMap_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()

            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer':
                        detail['NuscMap_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]
                    elif metric == 'iou':
                        if thr == 0.5 or thr == 0.75:
                            detail['NuscMap_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]

        return detail


    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name], metric=metric)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, metric=metric)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict




