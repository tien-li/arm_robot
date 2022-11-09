import cv2
import numpy as np
import matplotlib.path as mplPath
import rospy
from scipy.stats import mode
from scipy.spatial.distance import cdist
from copy import deepcopy
class BlockDetector:
    def __init__(self, intrinsic, extrinsic):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic

        red = {'color_name': 'red', 'color_code': (0, 0, 255), 'lower_limit': [175, 80, 50], 'upper_limit': [180, 255, 255]}
        # red_2 = {'color_name': 'red2', 'color_code': (0, 0, 255), 'lower_limit': [0, 100, 80], 'upper_limit': [10, 255, 255]}
        blue = {'color_name': 'blue', 'color_code': (255, 0, 0), 'lower_limit': [100, 180, 80], 'upper_limit': [110, 255, 255]}
        green = {'color_name': 'green', 'color_code': (0, 255, 0), 'lower_limit': [40, 80, 50], 'upper_limit': [80, 255, 255]}
        yellow = {'color_name': 'yellow', 'color_code': (0, 255, 255), 'lower_limit': [22, 80, 50], 'upper_limit': [35, 255, 255]}
        orange = {'color_name': 'orange', 'color_code': (0, 128, 255), 'lower_limit': [10, 80, 100], 'upper_limit': [20, 255, 255]}
        purple = {'color_name': 'purple', 'color_code': (255, 0, 128), 'lower_limit': [110,50, 40], 'upper_limit': [140, 255, 255]}
        # pink = {'color_name': 'pink', 'color_code': (255, 0, 255), 'lower_limit': [140, 0, 150], 'upper_limit': [175, 200, 255]}
        self.color_list = [red, blue, green, yellow, orange, purple]

        self.filtered_boxes = []
        self.refined_filtered_boxes = []
        self.filtered_boxes_centroids = []
        self.block_contours_dict = {}
        self.temp_block_dict = {}
        self.block_pos_color_dict = {}
        self.colors_dict = {}
        self.orientations_dict = {}
        self.crop_params = [0, -1, 0, -1]       # [row1, row2, col1, col2]

        self.small_filtered_boxes = []
        self.small_refined_filtered_boxes = []
        self.small_filtered_boxes_centroids = []
        self.small_block_pos_color_dict = {}
        self.block_sizes_dict_temp = {}
        self.block_sizes_dict = {}


    def depth_map_contour_detect(self):
        self.depth_img = cv2.fastNlMeansDenoising(self.depth_img, templateWindowSize=7, searchWindowSize=21, h=0)
        cv2.imwrite('thresh_img_cropped_pre.png', self.depth_img)
        ret, thresh = cv2.threshold(self.depth_img, 196, 255, 0)
        cv2.imwrite('threshold.png', thresh)
        ret_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        big_contours = [i for i in contours if i.shape[0] > 110 and i.shape[0] <= 600]
        small_contours = [i for i in contours if i.shape[0] > 50 and i.shape[0] <= 110]
        # print('Number of Big contours', len(big_contours))
        # print('Number of Small contours', len(small_contours))
        self.filtered_boxes, self.refined_filtered_boxes, self.refined_filtered_boxes_sizes, self.filtered_boxes_centroids = self.filter_contours(big_contours, 110, 600)
        self.small_filtered_boxes, self.small_refined_filtered_boxes, self.small_refined_filtered_boxes_sizes, self.small_filtered_boxes_centroids = self.filter_contours(small_contours, 50, 110)
        filt_box_add = []
        ref_filt_box_add = []
        filtered_boxes_centroids_add = []
        i_to_del = []

        for i in range(len(self.refined_filtered_boxes)):
            if self.refined_filtered_boxes_sizes[i] <= 110:
                # filt_box_add.append(self.filtered_boxes[i])
                # ref_filt_box_add.append(self.refined_filtered_boxes[i])
                # self.small_refined_filtered_boxes_sizes.append(self.refined_filtered_boxes_sizes[i])
                # self.small_filtered_boxes_centroids.append(self.filtered_boxes_centroids[i])
                i_to_del.append(i)


        # self.small_refined_filtered_boxes = np.concatenate((self.small_refined_filtered_boxes, self.refined_filtered_boxes[i_to_del]), axis=0)


        try:
            if len(i_to_del) > 0:
                self.small_filtered_boxes_centroids = np.concatenate((self.small_filtered_boxes_centroids, self.filtered_boxes_centroids[i_to_del]), axis=0)

            for i in i_to_del:
                self.small_refined_filtered_boxes.append(self.refined_filtered_boxes[i])
            new_refined_filtered_boxes = []
            for i in range(len(self.refined_filtered_boxes)):
                if i not in i_to_del:
                    new_refined_filtered_boxes.append(self.refined_filtered_boxes[i])
            self.refined_filtered_boxes = new_refined_filtered_boxes
            self.filtered_boxes_centroids = np.delete(self.filtered_boxes_centroids, i_to_del, axis=0)
        except:
            pass
        # print("Number of Big block centroids: ", len(self.filtered_boxes_centroids))
        # print("Number of small block centroids: ", len(self.small_filtered_boxes_centroids))

    def filter_contours(self, contours, minSize, maxSize):
        boxes = []
        contours_draw = []
        filtered_boxes = []
        refined_filtered_boxes = []
        filtered_boxes_centroids = []
        refined_filtered_boxes_sizes = []

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, .04 * peri, True)
            if len(approx) >= 4:
                contours_draw.append(contour)
                minRect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(minRect)
                box = np.intp(box)
                area = .5*np.abs(np.dot(box[:, 0], np.roll(box[:, 1], 1)) - np.dot(box[:, 1], np.roll(box[:, 0], 1)))
                boxes.append((area, box))

        boxes = sorted(boxes, reverse=True, key=lambda a: a[0])

        for i, (area, box_1) in enumerate(boxes):
            okay = True
            box_1_p = mplPath.Path(box_1, closed=False)
            for area, box_2 in boxes[i+1: ]:
                box_2_p = mplPath.Path(box_2, closed=False)
                if box_1_p.contains_path(box_2_p):
                    okay = False
                    break
            if okay:
                filtered_boxes.append(box_1)
        # print('Number of boxes, before and after:', len(boxes), len(filtered_boxes))

        for filtered_box in filtered_boxes:
            new_img = np.zeros_like(self.depth_img )
            cv2.drawContours(new_img, [filtered_box], 0,1, thickness=-1)
            im_2_filtered_box = self.depth_img  * new_img
            max_x = filtered_box[:, 0].max()
            min_x = filtered_box[:, 0].min()
            max_y = filtered_box[:, 1].max()
            min_y = filtered_box[:, 1].min()
            cropped_box = im_2_filtered_box[min_y:max_y, min_x:max_x]
            mode_img = mode(mode(cropped_box[cropped_box != 0], axis=0).mode, axis=None).mode[0]
            # mode_img = mode(cropped_box, axis=None).mode[0]
            second_min = np.amin(cropped_box[cropped_box != 0])
            im_2_filtered_box[im_2_filtered_box == second_min] = second_min - 50
            im_2_filtered_box[im_2_filtered_box == second_min + 1] = second_min - 50
            im_2_filtered_box[im_2_filtered_box == second_min + 2] = second_min - 50
            im_2_filtered_box[im_2_filtered_box == second_min + 3] = second_min - 50
            im_2_filtered_box[im_2_filtered_box == second_min + 4] = second_min - 50
            im_2_filtered_box[im_2_filtered_box == second_min + 5] = second_min - 50
            im_2_filtered_box[im_2_filtered_box == second_min + 6] = second_min - 50
            # pass_in = mode_img
            # if mode_img == 0:
            # cv2.imwrite('crop_' + str(max_x) + '.png', im_2_filtered_box)
            ret, thresh = cv2.threshold(im_2_filtered_box, second_min - 48, 255, 0)
            # cv2.imwrite('thresh_inner' + str(max_x) + '.png', thresh)
            ret_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            contours = [i for i in contours if i.shape[0] > 50]
            cur_min_ind = 0
            if len(contours) > 1:
                cur_min_ind = 0
                for i in range(len(contours)):
                    if contours[i].shape[0] < contours[cur_min_ind].shape[0]:
                        cur_min_ind = i
            if len(contours) > 0:
                contour = contours[cur_min_ind]
                refined_filtered_boxes_sizes.append(contours[cur_min_ind].shape[0])
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, .04 * peri, True)
                if len(approx) >= 4:
                    minRect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(minRect)
                    box = np.intp(box)
                    area = .5*np.abs(np.dot(box[:, 0], np.roll(box[:, 1], 1)) - np.dot(box[:, 1], np.roll(box[:, 0], 1)))
                    refined_filtered_boxes.append(box)
        for filtered_box in refined_filtered_boxes:
            box_centroid = filtered_box.mean(axis=0)
            filtered_boxes_centroids.append(box_centroid)
            cv2.drawContours(self.depth_img, [filtered_box], 0, 3)
            self.block_contours_dict[(box_centroid[0], box_centroid[1])] = filtered_box

        filtered_boxes_centroids = np.array(filtered_boxes_centroids)
        return filtered_boxes, refined_filtered_boxes, refined_filtered_boxes_sizes, filtered_boxes_centroids

    def draw_contours(self, mask, image, color_name, color_code):
        ret_img, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        box_list = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(image, color_name, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            color_code)

                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, .04 * peri, True)
                if len(approx) >= 2:
                    minRect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(minRect)
                    box = np.intp(box)
                    box_list.append(box)
        for box in box_list:
            # cv2.drawContours(self.img, [box], 0, 3)
            centroid = box.mean(axis=0)
            self.temp_block_dict[(centroid[0], centroid[1])] = color_name
        return image

    def nearest_centroid(self, centroid):
        centroid = np.array([[centroid[0], centroid[1]]])
        if self.filtered_boxes_centroids.shape[0] > 0:
            distance_to_big_blocks = np.linalg.norm(self.filtered_boxes_centroids - centroid, axis=1)
            id_big = np.argmin(distance_to_big_blocks)
            shortest_dist_big = distance_to_big_blocks[id_big]
        else:
            shortest_dist_big = None

        if self.small_filtered_boxes_centroids.shape[0] > 0:
            distance_to_small_blocks = np.linalg.norm(self.small_filtered_boxes_centroids - centroid, axis=1)
            id_small = np.argmin(distance_to_small_blocks)
            shortest_dist_small = distance_to_small_blocks[id_small]
        else:
            shortest_dist_small = None

        if shortest_dist_big is not None and shortest_dist_small is not None:
            if shortest_dist_big < shortest_dist_small:
                refined_centroid = self.filtered_boxes_centroids[id_big]
                return refined_centroid, "Large"
            else:
                refined_centroid = self.small_filtered_boxes_centroids[id_small]
                return refined_centroid, "Small"
        elif shortest_dist_big is not None:
            refined_centroid = self.filtered_boxes_centroids[id_big]
            return refined_centroid, "Large"
        elif shortest_dist_small is not None:
            refined_centroid = self.small_filtered_boxes_centroids[id_small]
            return refined_centroid, "Small"
        else:
            return None, None

    def nearest_centroid_world_coordinate(self, centroid):
        centroid = np.array([[centroid[0], centroid[1], centroid[2]]])
        centroids_array = np.array(list(self.orientations_dict.keys()))
        distance = np.linalg.norm(centroid-centroids_array, axis=1)
        id = np.argmin(distance)
        refined_centroid = centroids_array[id]
        return refined_centroid, distance[id]

    def nearest_centroid_world_coordinate_no_z(self, centroid):
        centroid = np.array([[centroid[0], centroid[1]]])
        centroids_array = np.array(list(self.orientations_dict.keys()))[:, :2]
        distance = np.linalg.norm(centroid-centroids_array, axis=1)
        id = np.argmin(distance)
        refined_centroid = centroids_array[id]
        return refined_centroid, distance[id]

    def compute_world_position(self):
        crop_row, _, crop_col, _ = self.crop_params
        bpcd = self.block_pos_color_dict.copy()
        bpcd.update(self.small_block_pos_color_dict)
        for centroid in bpcd:
            # print(centroid)
            x = centroid[0] + crop_col
            y = centroid[1] + crop_row
            z = self.raw_depth_img[int(y),int(x)]
            camera_coords = z * np.linalg.inv(self.intrinsic).dot(np.array([[x], [y], [1]]))
            camera_coords_homogeneous = np.concatenate((camera_coords, np.array([[1]])), axis=0)
            centroid_world_coords = np.linalg.inv(self.extrinsic).dot(camera_coords_homogeneous)
            Cw = centroid_world_coords.flatten()

            contour = self.block_contours_dict[centroid]
            pt_max_x_ind = np.argmax(contour[:, 0])
            pt_max_x = contour[pt_max_x_ind, 0]
            pt_max_y = contour[pt_max_x_ind, 1]

            p1_x = pt_max_x + crop_col
            p1_y = pt_max_y + crop_row
            z = self.raw_depth_img[p1_y][p1_x]
            camera_coords = z * np.linalg.inv(self.intrinsic).dot(np.array([[p1_x], [p1_y], [1]]))
            camera_coords_homogeneous = np.concatenate((camera_coords, np.array([[1]])), axis=0)
            p1_world_coords = np.linalg.inv(self.extrinsic).dot(camera_coords_homogeneous)

            pt_min_y_ind = np.argmin(contour[:, 1])
            pt_min_x = contour[pt_min_y_ind, 0]
            pt_min_y = contour[pt_min_y_ind, 1]

            p2_x = pt_min_x + crop_col
            p2_y = pt_min_y + crop_row
            z = self.raw_depth_img[p2_y][p2_x]
            camera_coords = z * np.linalg.inv(self.intrinsic).dot(np.array([[p2_x], [p2_y], [1]]))
            camera_coords_homogeneous = np.concatenate((camera_coords, np.array([[1]])), axis=0)
            p2_world_coords = np.linalg.inv(self.extrinsic).dot(camera_coords_homogeneous)
            diff = p1_world_coords - p2_world_coords
            angle = np.arctan(np.abs(diff[1]/diff[0])) #* 180 / np.pi

            size = self.block_sizes_dict_temp[centroid]

            if size == 'Large':
                z_offset = 20
                self.colors_dict[(Cw[0], Cw[1], Cw[2] - z_offset)] = self.block_pos_color_dict[centroid]
            else:
                z_offset = 8
                self.colors_dict[(Cw[0], Cw[1], Cw[2] - z_offset)] = self.small_block_pos_color_dict[centroid]


            self.block_sizes_dict[(Cw[0], Cw[1], Cw[2] - z_offset)] = size
            self.orientations_dict[(Cw[0], Cw[1], Cw[2] - z_offset)] = angle

    def detect_blocks(self, color_img, depth_img):
        self.crop_params = [110, -15, 240, -130]       #[row1, row2, col1, col2]
        row1, row2, col1, col2 = self.crop_params
        self.img = color_img
        self.img = self.img[row1:row2, col1:col2, :]
        self.depth_img = depth_img
        self.depth_img = self.depth_img[row1:row2, col1:col2].astype('uint8')

        self.filtered_boxes = []
        self.refined_filtered_boxes = []
        self.filtered_boxes_centroids = []
        self.block_contours_dict = {}
        self.temp_block_dict = {}
        self.block_pos_color_dict = {}
        self.colors_dict = {}
        self.orientations_dict = {}
        self.img = cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

        self.small_filtered_boxes = []
        self.small_refined_filtered_boxes = []
        self.small_filtered_boxes_centroids = []
        self.small_block_pos_color_dict = {}
        self.block_sizes_dict_temp = {}
        self.block_sizes_dict = {}
        self.raw_depth_img = deepcopy(depth_img)

        cv2.imwrite('color_img_cropped.png', self.img)
        self.depth_map_contour_detect()
        cv2.imwrite('thresh_img_cropped.png', self.depth_img)

        # Generate contour mask for color image
        contour_mask = np.zeros(self.img.shape, dtype=np.uint8)
        allowance = 0
        for filtered_box in self.refined_filtered_boxes:
            x, y, w, h = cv2.boundingRect(filtered_box)
            contour_mask[y+allowance:y+h-allowance, x+allowance:x+w-allowance, :] = np.ones_like(contour_mask[y+allowance:y+h-allowance, x+allowance:x+w-allowance, :], dtype=np.uint8)
        for filtered_box in self.small_refined_filtered_boxes:
            x, y, w, h = cv2.boundingRect(filtered_box)
            contour_mask[y+allowance:y+h-allowance, x+allowance:x+w-allowance, :] = np.ones_like(contour_mask[y+allowance:y+h-allowance, x+allowance:x+w-allowance, :], dtype=np.uint8)

        extract_boxes = self.img * contour_mask

        # color detection
        hsvFrame = cv2.cvtColor(extract_boxes, cv2.COLOR_BGR2HSV)

        kernal = np.ones((5, 5), "uint8")
        for key in self.color_list:
            lower_lim = np.array(key['lower_limit'], np.uint8)
            upper_lim = np.array(key['upper_limit'], np.uint8)
            mask = cv2.inRange(hsvFrame, lower_lim, upper_lim)
            self.img = self.draw_contours(mask, self.img, key['color_name'], key['color_code'])

        # print('temp_block_dict', self.temp_block_dict)
        for key in self.temp_block_dict:
            refined_centroid, size_ = self.nearest_centroid(key)
            if size_ == "Large":
                self.block_pos_color_dict[(refined_centroid[0], refined_centroid[1])] = self.temp_block_dict[key]
                self.block_sizes_dict_temp[(refined_centroid[0], refined_centroid[1])] = "Large"
            else:
                self.small_block_pos_color_dict[(refined_centroid[0], refined_centroid[1])] = self.temp_block_dict[key]
                self.block_sizes_dict_temp[(refined_centroid[0], refined_centroid[1])] = "Small"



        for box in self.refined_filtered_boxes:
            cv2.drawContours(self.img, [box], 0, 3)
        for box in self.small_refined_filtered_boxes:
            cv2.drawContours(self.img, [box], 0, 3)
        self.compute_world_position()
        for centroid in self.block_sizes_dict_temp:
            x, y = int(centroid[0]), int(centroid[1])
            cv2.putText(self.img, self.block_sizes_dict_temp[centroid][0], (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        # print(self.block_sizes_dict)

        # print("Position and colors of detected blocks: ", self.colors_dict)
        # print("Orientation of detected blocks: ", self.orientations_dict)
        cv2.imwrite('clor_img_detected_blocks.png', self.img)

        if len(self.orientations_dict) == 48:
            heatmap_array = np.zeros(48)
            total_error = 0
            left_grid_x, left_grid_y = np.meshgrid(np.linspace(-400, -100, 4), np.linspace(-75, 425, 6))
            right_grid_x, right_grid_y = np.meshgrid(np.linspace(100, 400, 4), np.linspace(-75, 425, 6))
            grid_x = np.concatenate((left_grid_x, right_grid_x), axis=1).reshape(-1, 1)
            grid_y = np.concatenate((left_grid_y, right_grid_y), axis=1).reshape(-1, 1)
            grid_x_y = np.concatenate((grid_x, grid_y), axis=1)
            grid_z = []
            small_centroids = [(-400, -75), (-300, 125), (-200, 125), (-100, 125), (-400, 225), (-200, 225),
                               (-300, 425), (100, 125), (300, 125), (200, 325), (100, 425)]
            for i in range(grid_x_y.shape[0]):
                centroid_i = grid_x_y[i]
                if tuple(centroid_i) in small_centroids:
                    grid_z.append(12.5)
                else:
                    grid_z.append(20)
            grid_z = np.array(grid_z).reshape(48, 1)
            grid = np.concatenate((grid_x_y, grid_z), axis=1)

            centroids = sorted(list(self.orientations_dict.keys()))
            for i in range(len(centroids)):
                if self.block_sizes_dict[centroids[i]] == 'Small':
                    # The stored centroid z values are offset from the actual z values to help account for gravity effects in the motion planner.
                    # This rectifies that for error calculation
                    pred_z_offset = -4.5
                else:
                    pred_z_offset = 0
                centroid_z_offset = np.array([[centroids[i][0], centroids[i][1], centroids[i][2] + pred_z_offset]])
                closest_centroid = np.argmin(cdist(centroid_z_offset, grid))
                error = np.linalg.norm(np.array(centroids[i]) - grid[closest_centroid])
                heatmap_array[closest_centroid] = error
                total_error += error

            print(total_error/48)
            print(heatmap_array.reshape(6, 8))
            print(grid_x_y)

        return self.img


if __name__ == '__main__':
    color_img = cv2.imread('block_detect_in/color_3.png', cv2.IMREAD_COLOR)
    depth_img = cv2.imread('block_detect_in/depth_3.png', cv2.COLOR_BGR2RGB)
    intrinsic_matrix = np.array([[987.18762 ,   0.      , 630.97839 ],
                    [  0.      , 983.123265, 367.15032 ],
                    [  0.      ,   0.      ,   1.      ]])
    extrinsic_matrix = np.array([[1, 0, 0, 60],
                    [0, -1, 0, 180],
                    [0 , 0, -1, 970],
                    [0 , 0, 0, 1]], dtype=np.float64)
    block_detector = BlockDetector(intrinsic_matrix, extrinsic_matrix)
    color_img = block_detector.detect_blocks(color_img, depth_img)


    """
    IMPORTANT STUFF:
     - fix slant to detect small blocks
     - take care that you can distinguish between two small blocks stacked and a large block
    """
