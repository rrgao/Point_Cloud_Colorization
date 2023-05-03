import numpy as np
import logging
import torch
import torch.utils.data
from scipy.linalg import expm, norm
from easydict import EasyDict as edict

import MinkowskiEngine as ME
import lib.transforms as t
from config_scannet_2 import get_config
import os

def collate_pair_fn(list_data):
  coords, feats, xyz_down, rgb_down = list(zip(*list_data))

  xyz_batch = []
  rgb_batch = []
  len_batch = []

  batch_id = 0
  curr_start_inds = np.zeros((1, 2))

  # transfrom x into torch tensor
  def to_tensor(x):
    if isinstance(x, torch.Tensor): return x
    elif isinstance(x, np.ndarray): return torch.from_numpy(x)
    else: raise ValueError(f'Can not convert to torch tensor, {x}')

  for batch_id, _ in enumerate(coords):
    N0 = coords[batch_id].shape[0]
    xyz_batch.append(to_tensor(xyz_down[batch_id]))
    rgb_batch.append(to_tensor(rgb_down[batch_id]))
    len_batch.append([N0])
    # Move the head
    curr_start_inds[0, 0] += N0

  coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats)
  # Concatenate all lists
  xyz_batch = torch.cat(xyz_batch, 0).float()
  rgb_batch = torch.cat(rgb_batch, 0).float()
  return {
      'pcd': xyz_batch,
      'rgb': rgb_batch,
      'sinput_C': coords_batch,
      'sinput_F': feats_batch.float(),
      'len_batch': len_batch
  }

# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

def sample_random_trans(pcd, randg, rotation_range=360):
  T = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T[:3, :3] = R
  T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
  return T


class SDataset(torch.utils.data.Dataset):
  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               mconfig=None):
    self.phase = phase
    self.features = []
    self.transform = transform
    self.voxel_size = mconfig.voxel_size

    self.random_scale = random_scale
    self.min_scale = mconfig.min_scale
    self.max_scale = mconfig.max_scale
    self.random_rotation = random_rotation
    self.rotation_range = mconfig.rotation_range
    self.randg = np.random.RandomState()
    if manual_seed:
      self.reset_seed()

  def reset_seed(self, seed=0):
    logging.info(f"Resetting the data loader seed to {seed}")
    self.randg.seed(seed)

  def apply_transform(self, pts, trans):
    R = trans[:3, :3]
    T = trans[:3, 3]
    pts = pts @ R.T + T
    return pts

  def __len__(self):
    return len(self.features)

class TrainDataset(SDataset):
  DATA_DIRS = {
    'train': '/disk1/rongrong/scan/train_xyz_rgb_lab_geo_0.025/'
  }
  def __init__(self,
               phase,
               transform=None,
               random_rotation=True,
               random_scale=True,
               manual_seed=False,
               mconfig=None):
    SDataset.__init__(self, phase, transform, random_rotation, random_scale, manual_seed, mconfig)
    assert phase == 'train', "Supports only the train set."
    list_file = os.listdir(self.DATA_DIRS[phase])
    list_file.sort()
    for k, file in enumerate(list_file):
      temp = np.loadtxt(os.path.join(self.DATA_DIRS[phase], file), delimiter=' ', dtype='float32')
      temp1 = np.loadtxt(os.path.join("/disk1/rongrong/scan/semantic_0.025/", file), delimiter=' ', dtype='float32')
      fea = np.concatenate((temp, temp1), axis=1)
      self.features.append(fea)

  def __getitem__(self, idx):
    data_idx = self.features[idx]
    xyz = np.asarray(data_idx)[:, 0:3]
    rgb = np.asarray(data_idx)[:, 3:6]
    #rgb =  np.concatenate((rgb_, rgb_, rgb_), axis=1)
    #lab = np.asarray(data_idx)[:, 3:6]
    #l = (lab[:, 0]/2).astype(np.int) # class label
    #l = lab[:, 0]
    geo_feature = np.asarray(data_idx)[:, 6:22]
    seg_feature = np.asarray(data_idx)[:, 22:42]
    geo_f = np.concatenate((geo_feature, seg_feature), axis=1)
    #geo_f = np.expand_dims(geo_f, axis=1)

    ##########random scale
    # if self.random_scale and random.random() < 0.95:
    #   scale = self.min_scale + \
    #           (self.max_scale - self.min_scale) * random.random()
    #   xyz = scale*xyz
    ##########random rotate
    # if self.random_rotation:
    #   T = sample_random_trans(xyz, self.randg, self.rotation_range)
    #   xyz = self.apply_transform(xyz, T)
    # else:
    #   trans = np.identity(4)

    # Get feats and coords
    feats = []
    feats.append(geo_f)
    #feats.append(np.ones((len(xyz), 1)))
    feats = np.hstack(feats)
    corrds = xyz

    return (corrds, feats, xyz, rgb)


def make_data_loader(mconfig, phase, batch_size, num_threads=0, shuffle=None):

  assert phase in ['train', 'trainval', 'val', 'test']
  if shuffle is None:
    shuffle = phase != 'test'

  use_random_scale = False
  use_random_rotation = False
  transforms = []
  if phase in ['train', 'trainval']:
    use_random_rotation = mconfig.use_random_rotation
    use_random_scale = mconfig.use_random_scale
    transforms = [t.Jitter()]

  dset = TrainDataset(
    phase=phase,
    transform=t.Compose(transforms),
    random_scale=use_random_scale,
    random_rotation=use_random_rotation,
    mconfig=mconfig)

  loader = torch.utils.data.DataLoader(
    dset,
    batch_size = batch_size,
    shuffle = shuffle,
    num_workers = num_threads,
    collate_fn=collate_pair_fn,
    pin_memory=False,
    drop_last=True)

  return loader

if __name__ == "__main__":

  mconfig = get_config()
  dconfig = vars(mconfig)
  mconfig = edict(dconfig)
  train_loader = make_data_loader(mconfig=mconfig,
                                  phase='train',
                                  batch_size=mconfig.batch_size,
                                  num_threads=mconfig.train_num_thread)
  print(train_loader)













