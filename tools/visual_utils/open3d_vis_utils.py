"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

box_textmap = ['Undefined', 'Car', 'Pedestrian', 'Cyclist']


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, thold=0,point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    if isinstance(ref_scores, torch.Tensor):
        ref_scores = ref_scores.cpu().numpy()
    if isinstance(ref_labels, torch.Tensor):
        ref_labels = ref_labels.cpu().numpy()


    if thold != 0:
        i = 0

        while True:

            if ref_scores[i] < thold:
                
                ref_boxes = np.delete(ref_boxes,i, 0)
                ref_scores = np.delete(ref_scores,i)
                ref_labels = np.delete(ref_labels,i)
                
                i -= 1

            i += 1

            if i == len(ref_boxes):
                break

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes, score, label):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    rot_text = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles + 3/8*np.pi) 
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    text_pos = box3d.get_box_points()[6]

    text = str(round(float(score)*100,1)) + "%"

    score_3d = text_3d(text, text_pos, rot_text)

    text_pos = box3d.get_box_points()[5]

    text = box_textmap[label]

    class_3d = text_3d(text, text_pos, rot_text)

    return line_set, score_3d, class_3d

def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, scores=None):
    for i in range(gt_boxes.shape[0]):
        line_set, score_3d, class_3d = translate_boxes_to_open3d_instance(gt_boxes[i], scores[i], ref_labels[i])

        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)
        vis.add_geometry(score_3d)
        vis.add_geometry(class_3d)

    return vis

def text_3d(text, pos, direction, density=3, font='/lhome/fimilak/Documents/OpenPCDet/tools/visual_utils/Arial.ttf', font_size=20):
    """
    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """

    direction = [[1,0,0],[0,1,0],[0,0,1]]

    font_obj = ImageFont.truetype(font, font_size * density)

    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255,255,255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0,0,0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = open3d.geometry.PointCloud()
    #pcd.colors = open3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.colors = open3d.utility.Vector3dVector([(1,0,0) for _ in range(len(img_mask))])
    pcd.points = open3d.utility.Vector3dVector(indices / 100 / density)

    #raxis = np.cross([0.0, 0.0, 1.0], direction)
    #if np.linalg.norm(raxis) < 1e-6:
    #    raxis = (0.0, 0.0, 1.0)
    #trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
    #         Quaternion(axis=direction, degrees=degree)).transformation_matrix

    trans = np.append(np.append(direction,np.asarray(pos).reshape((3,1)), axis=1),[[0,0,0,1]],axis=0)
    
    pcd.transform(trans)

    return pcd
