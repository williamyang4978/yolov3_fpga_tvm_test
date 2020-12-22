# To be able to target the Vitis-AI cloud DPUCZDX8G target we first have to import the target in PyXIR. i
# This PyXIR package is the interface being used by TVM to integrate with the Vitis-AI stack. Additionaly, i
# import the typical TVM and Relay modules and the Vitis-AI contrib module inside TVM.
import sys
import numpy as np

import pyxir
import pyxir.contrib.target.DPUCZDX8G

import tvm
import tvm.relay as relay
from tvm.contrib.target import vitis_ai
from tvm.contrib import utils, graph_runtime
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.contrib.vitis_ai import annotation
import colorsys
import random
from PIL import Image

# After importing a convolutional neural network model using the usual Relay API's, 
# annotate the Relay expression for the given Vitis-AI DPU target and partition the graph.
import cv2
from tvm.relay.frontend.tensorflow_parser import TFParser
#from tensorflow.contrib import decent_q

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def draw_bbox(image, bboxes, classes, show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    valid_scale = [0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def image_preporcess(image, target_size, gt_boxes=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


input_size = 416
original_image = cv2.imread('horses.jpg')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...]
image_data = image_data.astype(np.float32)

CHECKPOINT = './saved/'
layout = "NCHW"

data_shape = image_data.shape
shape_dict = {'input/input_data': image_data.shape}
dtype_dict = {'input/input_data': 'uint8'}

return_elements = ["pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
parser = TFParser(CHECKPOINT)
graph_def = parser.parse()

mod, params = relay.frontend.from_tensorflow(graph_def,
                                             layout=layout,
                                             shape=shape_dict,
                                             outputs=return_elements)

#category_file = os.path.join(CHECKPOINT, 'category.txt')
classes = read_class_names('category.txt')
num_classes = len(classes)

tvm_target = 'llvm'
target ='DPUCZDX8G-zcu104'
net = mod["main"]
mod = tvm.IRModule.from_expr(net)

mod["main"] = bind_params_by_name(mod["main"], params)
mod = relay.transform.InferType()(mod)
mod = annotation(mod, params, target)
mod = relay.transform.MergeCompilerRegions()(mod)
mod = relay.transform.PartitionGraph()(mod)


# build the TVM runtime library for executing the model

#from tvm.contrib import util
temp = utils.tempdir()
export_rt_mod_file = temp.relpath("vitis_ai.rtmod")

with tvm.transform.PassContext(opt_level=3, config= {'relay.ext.vitis_ai.options.target': target,
                                                     'relay.ext.vitis_ai.options.export_runtime_module': export_rt_mod_file}):
   lib = relay.build(mod, tvm_target, params=params)




# We will quantize and compile the model for execution on the DPU using on-the-fly quantization on the host machine. 
# This makes use of TVM inference calls (module.run) to quantize the model on the host with the first N inputs.

m = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
dtype = 'float32'
# First N (default = 128) inputs are used for quantization calibration and will
# be executed on the CPU
# This config can be changed by setting the 'PX_QUANT_SIZE' (e.g. export PX_QUANT_SIZE=64)
for i in range(128):
   m.set_input('input/input_data', tvm.nd.array(image_data.astype(dtype)))
  # module.set_input(input_name, inputs[i])
   m.run()


# get outputs
print(m.get_output(0).shape)
print(m.get_output(1).shape)
print(m.get_output(2).shape)


pred_sbbox = m.get_output(0).asnumpy()
pred_mbbox = m.get_output(1).asnumpy()
pred_lbbox = m.get_output(2).asnumpy()

pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
bboxes = nms(bboxes, 0.45, method='nms')
image = draw_bbox(original_image, bboxes, classes)
image = Image.fromarray(image)

first_name, last_name = 'horses.jpg'.split(".")
predict_image_name = first_name + "_predict." + last_name
image.save(predict_image_name)


# Save the TVM lib module so that the Vitis-AI runtime module will also be exported (to the 'export_runtime_module' path we previously passed as a config).

#from tvm.contrib import util

temp = utils.tempdir()
#lib.export_library(temp.relpath("tvm_lib.so"))
lib.export_library("tvm_lib.so")

# Export lib for aarch64 target
#tvm_target = tvm.target.arm_cpu('DPUCZDX8G-zcu104')
tvm_target = tvm.target.arm_cpu('ultra96')
lib_kwargs = {
     'fcompile': contrib.cc.create_shared,
     'cc': "/usr/aarch64-linux-gnu/bin/ld"
}

with tvm.transform.PassContext(opt_level=3,
                               config={'relay.ext.vitis_ai.options.load_runtime_module': export_rt_mod_file}):
     lib_arm = relay.build(mod, target=tvm_target, params=params)
     #lib_arm = relay.build_module.build(mod, target=tvm_target, params=params)


lib_dpuv2.export_library('tvm_dpu_arm.so', **lib_kwargs)

