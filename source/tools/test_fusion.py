from ensemble_boxes import *

boxes_list = [[
    [0.7328253038194444, 0.16287770414557068, 0.9605020616319445, 0.25692220532331345],
    [0.06580436197916667, 0.0, 0.6762463650173611, 0.15796390910005365],
    [0.053212456597222235, 0.18241040798727534, 0.6800546332465278, 0.2816332837542751],
    [0.6868729112413194, 0.00019972641580606743, 0.997143581814236, 0.12560499461423685],
],[
    [0.06488888888888888, 0.16051502145922747, 0.9644444444444444, 0.28412017167381975],
    [0.06933333333333333, 0.0017167381974248926, 0.9857777777777778, 0.13819742489270387],
],[
    [0.056469999999999965, 0.16358, 0.9607699999999999, 0.2853],
    [0.07554000000000005, 0.0001300000000000051, 0.99524, 0.15549000000000002],
]]
scores_list = [[0.8714041113853455, 0.859368622303009, 0.8215535283088684, 0.2967867851257324], [0.6884765625, 0.7529296875], [0.9530216455459595, 0.9509159922599792]]
labels_list = [[0, 0, 0, 0], [0, 0], [0, 0]]
weights = [1, 1, 1]

iou_thr = 0.5
skip_box_thr = 0.1
sigma = 0.1

boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)

print(f"mns: {boxes} {scores} {labels}")

boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)

print(f"soft_nms: {boxes} {scores} {labels}")

boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

print(f"non_maximum_weighted: {boxes} {scores} {labels}")

boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)

print(f"weighted_boxes_fusion: {boxes} {scores} {labels}")
