import numpy as np
import open3d as o3d
from skimage import color
import os

stage = "train"
epoches = [50]

def save_screenshots():
    for epoch in epoches:
        dir_color = "./results/" + stage + "/" + stage + "_" + str(epoch)
        dir_gray = "./results/" + stage + "/" + stage + "_" + str(epoch) + "_gray"
        dir_original = "./results/" + stage + "/" + stage + "_original"
        dir_original_gray = "./results/" + stage + "/" + stage + "_down_gray"

        # dirs = [dir_original, dir_original_gray, dir_color, dir_gray]
        dirs = [dir_color]
        for dir in dirs:
            plys = [b for b in os.listdir(dir) if b[0] != "."]
            plys.sort()
            for ply in plys:
                name = ply.split(".ply")[0]
                snapshot = os.path.join(dir, name + ".png")
                if not os.path.exists(snapshot):
                    pcd = o3d.io.read_point_cloud(os.path.join(dir, ply))
                    vis = o3d.visualization.Visualizer()
                    nps = len(np.array(pcd.points))
                    print(snapshot, nps)
                    delta_x = np.max(np.array(pcd.points)[:, 0]) - np.min(np.array(pcd.points)[:, 0])
                    delta_y = np.max(np.array(pcd.points)[:, 1]) - np.min(np.array(pcd.points)[:, 1])
                    delta_z = np.max(np.array(pcd.points)[:, 2]) - np.min(np.array(pcd.points)[:, 2])
                    voxel = delta_x * delta_y * delta_z / nps
                    if voxel > 40:
                        w = 1000
                        h = 800
                    elif voxel > 30:
                        print("> 10000 " + str(voxel))
                        w = 800
                        h = 600
                    elif voxel > 10:
                        print("> 8000 " + str(voxel))
                        w = 600
                        h = 400
                    else:
                        w = 400
                        h = 200
                    print(snapshot + " " + str(w) + " " + str(h))
                    vis.create_window(name, width=w, height=h)
                    vis.add_geometry(pcd)
                    vis.capture_screen_image(snapshot, do_render=True)
                    # vis.destroy_window()

def save_screenshots_mm():
    #dirs = "./test_try/ResUNet_k5/"
    dirs = "/Users/rongronggao/Downloads/16_60/"
    plys = [b for b in os.listdir(dirs) if (b[0] != "." and not b.endswith(".png"))]
    plys.sort()
    for ply in plys:
        name = ply.split(".ply")[0]
        snapshot = os.path.join(dirs, name + ".png")
        if not os.path.exists(snapshot):
            pcd = o3d.io.read_point_cloud(os.path.join(dirs, ply))
            vis = o3d.visualization.Visualizer()
            nps = len(np.array(pcd.points))
            print(snapshot, nps)
            w, h = 800, 400
            vis.create_window(name, width=w, height=h)
            vis.add_geometry(pcd)
            #vis.capture_screen_image(snapshot, do_render=True)


def vis_ss():
    dirs = "/Users/rongronggao/Downloads/FCGF-master/results/geo_div_bf/geo_div_single/"
    plys = [b for b in os.listdir(dirs) if (b[0] != "." and not b.endswith(".png"))]
    plys.sort()
    for ply in plys:
        pcd = o3d.io.read_point_cloud(os.path.join(dirs, ply))
        #p_c = np.concatenate((np.array(pcd.points), np.array(pcd.colors)), axis=1)
        pc_cm = colorfulness_metric(np.array(pcd.colors))
        print(pc_cm)


def test_effect():
    dirs = "/Users/rongronggao/Downloads/FCGF-master/results/ori_vis/val_vis/"
    dirs_m = "/Users/rongronggao/Downloads/FCGF-master/results/ori_vis/val_mean_vis/"
    dirs_lab = "/Users/rongronggao/Downloads/FCGF-master/results/ori_vis/val_l_vis/"
    plys = [b for b in os.listdir(dirs) if (b[0] != "." and not b.endswith(".png"))]
    plys.sort()
    for ply in plys:
        name = ply.split(".ply")[0]
        snapshot = os.path.join(dirs, name + ".png")
        if not os.path.exists(snapshot):
            pcd = o3d.io.read_point_cloud(os.path.join(dirs, ply))
            point = np.array(pcd.points)
            rgb = np.array(pcd.colors)
            tmp = np.sum(rgb, axis=1)
            tmp = tmp[:, np.newaxis]
            rgb_ = np.zeros((len(point), 3), dtype=np.float32)
            rgb_[:, 0] = tmp[:,0]
            rgb_[:, 1] = tmp[:,0]
            rgb_[:, 2] = tmp[:,0]

            vis_pcd = o3d.geometry.PointCloud()
            vis_pcd.points = o3d.utility.Vector3dVector(np.asarray(point))
            vis_pcd.colors = o3d.utility.Vector3dVector(np.asarray(rgb_))
            o3d.io.write_point_cloud(os.path.join(dirs_m, ply), vis_pcd)

            rgb1 = rgb.reshape((rgb.shape[0], 1, 3))
            lab = color.rgb2lab(rgb1).reshape((rgb.shape[0], 3))

            lab_ = np.zeros((len(point), 3), dtype=np.float32)
            lab_[:, 0] = lab[:,0]
            lab_[:, 1] = lab[:,0]
            lab_[:, 2] = lab[:,0]

            vis_pcd1 = o3d.geometry.PointCloud()
            vis_pcd1.points = o3d.utility.Vector3dVector(np.asarray(point))
            vis_pcd1.colors = o3d.utility.Vector3dVector(np.asarray(lab_/256.0))
            o3d.io.write_point_cloud(os.path.join(dirs_lab, ply), vis_pcd)





def main():
    #dirs = "./results/af/100/"
    #dirs = "./re_train/ResUNetBN2C_sl1_geo/"
    #dirs = "./test_xyz_rgb_0.05_12000_vis/"
    #dirs = "./test_test_l1/20/"
    #dirs = "/Users/rongronggao/cm_t/cm_t/"
    dirs = "/Users/rongronggao/Downloads/FCGF-master/results/seg_geo_IRT_ad/"
    plys = [b for b in os.listdir(dirs) if (b[0] != ".") and b.endswith(".ply")]#74
    plys.sort()
    for ply in plys:
    #for i in range(0, 1):
        #ply = plys[i]
        if not ply.endswith(".png"):
            print(os.path.join(dirs, ply))
            pcd = o3d.io.read_point_cloud(os.path.join(dirs, ply))
            o3d.visualization.draw_geometries([pcd])


#main()
save_screenshots_mm()
#vis_ss()
#test_effect()