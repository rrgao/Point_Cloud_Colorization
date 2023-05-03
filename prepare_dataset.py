import os
import numpy as np
import open3d as o3d

import torch
import MinkowskiEngine as ME
from model.resunet import ResUNetBN2C

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def extract_geo_feature(list, des_dir):
    voxel_size = 0.001

    #network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = "./pretrain_model/ResUNetBN2C-16feat-3conv.pth"
    model = ResUNetBN2C(1, 16, normalize_feature=False, conv1_kernel_size=3, D=3)
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)

    #for i, scene_id in enumerate(list):
    for k, file in enumerate(list):
        # ply_path = p_dir + scene_id + "/" + scene_id +"_vh_clean_2.ply"
        # pcd = o3d.io.read_point_cloud(ply_path) #(N, 6)
        # points = np.array(pcd.points)
        # rgb = np.array(pcd.colors)

        temp = np.loadtxt(file, delimiter=' ', dtype='float32')
        points = np.asarray(temp)[:, 0:3]
        rgb = np.asarray(temp)[:, 3:6]

        feats = []
        feats.append(np.ones((len(points), 1)))
        feats = np.hstack(feats)

        coords = np.floor(points / voxel_size)
        inds = ME.utils.sparse_quantize(coords, return_index=True)
        coords = coords[inds]
        coords = ME.utils.batched_coordinates([coords])
        feats = feats[inds]

        points_down = (np.floor(points / voxel_size))[inds]
        rgb_down = rgb[inds]

        feats = torch.tensor(feats, dtype=torch.float32)
        coords = torch.tensor(coords, dtype=torch.int32)
        stensor = ME.SparseTensor(feats, coords=coords).to(0)

        # # feature out
        feature = model(stensor).F
        feature = np.array(feature.detach().cpu().numpy())

        out_path = os.path.join(out_dir, str(k) + ".txt")

        with open(out_path, "w") as out:
            for k in range(len(points_down)):
                # print(points_down[k], feature[k])
                # print(points_down[k], feature[k])
                point = points_down[k]
                fea = feature[k]
                lab_ = rgb_down[k]
                out.write(str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + " ")
                out.write(str(lab_[0]) + " " + str(lab_[1]) + " " + str(lab_[2]) + " ")
                out.write(str(fea[0]) + " " + str(fea[1]) + " " + str(fea[2]) + " " + str(fea[3]) + " " + str(fea[4])
                          + " " + str(fea[5]) + " " + str(fea[6]) + " " + str(fea[7]) + " " + str(fea[8])
                          + " " + str(fea[9]) + " " + str(fea[10]) + " " + str(fea[11]) + " " + str(fea[12])
                          + " " + str(fea[13]) + " " + str(fea[14]) + " " + str(fea[15]))
                out.write("\n")
        print(out_path, " saved!")


def save_vis(list, in_dir):
    for k, name in enumerate(list):
        txt_file_path = os.path.join(in_dir, name+".txt")
        print(txt_file_path)
        temp = np.loadtxt(txt_file_path, delimiter=' ', dtype='float32')
        point = temp[:, 0:3]
        color = temp[:, 3:6]

        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(np.asarray(point))
        vis_pcd.colors = o3d.utility.Vector3dVector(np.asarray(color))
        o3d.io.write_point_cloud(os.path.join("/disk1/rongrong/scan/train_vis/", name + ".ply"), vis_pcd)

        print(name + ".ply")


if __name__=="__main__":
    p_dir = "/disk1/rongrong/scan/scans/"

    #stage = ["train", "val", "test"]
    stage = "train"
    #list_file = "./dataset/scannet_" + stage[0] + ".txt"
    list_file = "./dataset/scannet_train.txt"
    out_dir = "/disk1/rongrong/scan/train_vis/"
    in_dir = "/disk1/rongrong/scan/train_xyz_rgb_lab_geo_0.025/"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    list = []
    with open(list_file, "r") as listfile:
        for line in listfile.readlines():
            list.append(line.strip("\n"))
    list.sort()
    save_vis(list, in_dir)



