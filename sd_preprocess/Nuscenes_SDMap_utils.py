from nuscenes.utils.splits import create_splits_scenes

MAP = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
CLASS2LABEL = {
    'road_divider': 0, 
    'lane_divider': 0,
    'ped_crossing': 1, 
    'contours': 2,     
    'others': -1
}

NUM_CLASSES = 3
IMG_ORIGIN_H = 900
IMG_ORIGIN_W = 1600

# get the scenes of Nuscenes dataset
def get_scenes(version, is_train):
    split = {
        'v1.0-trainval': {True: 'train', False: 'val'},
        'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        'v1.0-test': {True: 'test', False: 'test'},
    }[version][is_train]

    return create_splits_scenes()[split]

# get the sample of Nuscenes dataset, used to set the dataloader's length
def get_samples(nusc,scenes):
    samples = [samp for samp in nusc.sample]
    scene_id=[]
    for samp in samples:
        scene_id.append(nusc.get('scene', samp['scene_token'])['name'])

    samples = [samp for samp in samples if 
                nusc.get('scene', samp['scene_token'])['name'] in scenes]

    samples.sort(key=lambda x: (x['scene_token'], x['timestamp'])) 

    return samples