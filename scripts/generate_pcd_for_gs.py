import numpy as np 
import open3d as o3d
import PIL 
import os
import cv2
import re
import struct

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale



def generate_pcd_for_gs(depth, color, K, extrinsic=None):
    '''
        depth: np.array of shape (H,W)
        color: np.array of shape (H,W,3)
        K: np.array of shape (3,3)
        extrinsic: np.array of shape (4,4), c2w
    '''
    # Generate point cloud
    depth = np.ascontiguousarray(depth.astype(np.float32))
    color = np.ascontiguousarray(color.astype(np.uint8))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(depth),
        depth_scale=1.0,
        depth_trunc=1000.0,
        convert_rgb_to_intensity=False)
    height, width = depth.shape
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width, height=height, fx=K[0,0], fy=K[1,1], cx=K[0,2], cy=K[1,2])    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    if extrinsic is not None:
        pcd.transform(extrinsic)
    # Transform point cloud to global frame
    #TODO 

    
    return pcd 

def write_colmap_binary_points3D(file_path, points):
    """
    Writes point cloud data to a COLMAP compatible points3D.bin file.

    Args:
        file_path (str): Path to the output points3D.bin file.
        points (numpy.ndarray): Array containing point data. 
                                Shape: (N, 8 + 2*M), where N is the number of points,
                                8 corresponds to (id, x, y, z, r, g, b, error), and
                                M is the number of views the point is seen in.
    """
    with open(file_path, 'wb') as f:
        f.write(struct.pack('<Q', len(points))) # Number of points

        for point in points:
            point_id = int(point[0])
            x, y, z = point[1:3+1]
            r, g, b = point[4:6+1]
            error = point[7]
            view_list = point[8:]

            f.write(struct.pack('<QdddBBBd', point_id, x, y, z, int(r), int(g), int(b), error))
            f.write(struct.pack('<Q', len(view_list)//2)) # Number of views
            for i in range(0, len(view_list), 2):
                 f.write(struct.pack('<II', int(view_list[i]), int(view_list[i+1])))
def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """

    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
        """Read and unpack the next bytes from a binary file.
        :param fid:
        :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
        :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
        :param endian_character: Any of {@, =, <, >, !}
        :return: Tuple of read and unpacked values.
        """
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))
        count = 0
        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            if error > 2.0 or track_length < 3:
                continue
            xyzs[count] = xyz
            rgbs[count] = rgb
            errors[count] = error
            count += 1
    xyzs = np.delete(xyzs, np.arange(count,num_points),axis=0)
    rgbs = np.delete(rgbs, np.arange(count,num_points),axis=0)
    errors = np.delete(errors, np.arange(count,num_points),axis=0)
    return xyzs, rgbs, errors

if __name__=='__main__':

    # seq = 'multi'
    seq = 'single'

    if seq == 'multi':
        near=0.5
        far=250.
        height = 576 
        width = 1024 
        scene = 'scan1'
        camera_extrinsics1 = np.loadtxt(f'example_imgs/multi/{scene}/000001.txt', skiprows=1, max_rows=3)
        R1 = camera_extrinsics1[:,0:3]
        T1 = camera_extrinsics1[0:3,3].reshape(3)
        camera_extrinsics2 = np.loadtxt(f'example_imgs/multi/{scene}/000002.txt', skiprows=1, max_rows=3)
        R2 = camera_extrinsics2[:,0:3]
        T2 = camera_extrinsics2[0:3,3].reshape(3)

        K = np.loadtxt(f'example_imgs/multi/{scene}/000001.txt', skiprows=7, max_rows=3)

        K[0,0] = K[0,0]/640*1024*4; K[1,1] = K[1,1]/512*576.*4; K[0,2] = K[0,2]/640*1024*4; K[1,2] = K[1,2]/512*576.*4
        K = K.astype(np.float32)

        image_l = PIL.Image.open(f'example_imgs/multi/{scene}/000001.jpg')
        image_l = image_l.resize((width,height),PIL.Image.Resampling.NEAREST) 
        image_l = np.array(image_l)#/255.

        image_r = PIL.Image.open(f'example_imgs/multi/{scene}/000002.jpg')
        image_r = image_r.resize((width,height),PIL.Image.Resampling.NEAREST) 
        image_r = np.array(image_r)#/255.

        depth_l,_ = read_pfm(f'example_imgs/multi/{scene}/000001.pfm' )
        depth_l = cv2.resize(depth_l, (width, height), interpolation=cv2.INTER_NEAREST)

        depth_r,_ = read_pfm(f'example_imgs/multi/{scene}/000001.pfm' )
        # new_width, new_height = 1024,576  # New dimensions
        depth_r = cv2.resize(depth_r, (width, height), interpolation=cv2.INTER_NEAREST)

    
        extrinsic_l = np.eye(4)
        extrinsic_l[:3,:] = camera_extrinsics1  # w2c 
        extrinsic_r = np.eye(4)
        extrinsic_r[:3,:] = camera_extrinsics2
        pcd_l = generate_pcd_for_gs(depth_l, image_l, K, np.linalg.inv(extrinsic_l) )
        pcd_r = generate_pcd_for_gs(depth_r, image_r, K, np.linalg.inv(extrinsic_r) )

        # pcd_l = generate_pcd_for_gs(depth_l, image_l, K, extrinsic_l )
        # pcd_r = generate_pcd_for_gs(depth_r, image_r, K, extrinsic_r )
        o3d.visualization.draw_geometries([pcd_l, ])

    elif seq == 'single':
        near=0.0001
        far=500.
        focal = 260.
        K = np.eye(3)
        K[0,0] = focal; K[1,1] = focal; K[0,2] = 1024./2; K[1,2] = 576./2
        new_width, new_height = 1024,576 

        image_path = 'example_imgs/single/000001.jpg'
        depth_path = 'example_imgs/single/depth/000001.npy'

        image_o = PIL.Image.open(image_path)
        image_o = image_o.resize((1024,576),PIL.Image.Resampling.NEAREST) 
        image = np.array(image_o)
        depth = np.load(depth_path).astype(np.float32)
        depth[depth < 1e-5] = 1e-5
        depth = 10000./depth 
        depth = np.clip(depth, near, far)
        # new_width, new_height = 1024,576  # New dimensions
        depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        pcd = generate_pcd_for_gs(depth, image, K)
        pcd = pcd.uniform_down_sample( 50)

        # o3d.visualization.draw_geometries([pcd, ])
        dataframes = []
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        for i in range(len(points)):
            data = np.concatenate([np.array([i]), points[i], colors[i]*255, [0.0], [0,0,1,1,2,2] ])
            dataframes.append(data)

        # import pdb; pdb.set_trace()
        write_colmap_binary_points3D('example_imgs/single/points3D.bin', dataframes)
        xyzs, rgbs, errors = read_points3D_binary('example_imgs/single/points3D.bin')
        # import pdb; pdb.set_trace()
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(xyzs)
        pcd1.colors = o3d.utility.Vector3dVector(rgbs/255.)
        o3d.visualization.draw_geometries([pcd, pcd1])






