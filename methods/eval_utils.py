import torch


def compute_eval_diffs(results_collection, coco_classes):
    h_before_total = 0
    recall_before_total = 0
    h_after_total = 0
    recall_after_total = 0
    h_intersect_total = 0
    recall_intersect_total = 0
    classes_to_count = dict()
    recall_lost_total = 0
    new_h_total = 0
    lost_h_total = 0
    recall_gained_total = 0
    h_free_images = [0, 0]

    for coco_class in coco_classes:
        classes_to_count[coco_class] = {
            "h_before": 0,
            "h_after": 0,
            "recall_before": 0,
            "recall_after": 0,
        }

    examples_good = []
    count = 0
    for coco_img in results_collection:
        count += 1
        h_before, recall_before, h_after, recall_after = results_collection[coco_img]
        for caption_word, coco_class in h_before:
            classes_to_count[coco_class]["h_before"] += 1
        for caption_word, coco_class in h_after:
            classes_to_count[coco_class]["h_after"] += 1

        for caption_word, coco_class in recall_before:
            classes_to_count[coco_class]["recall_before"] += 1
        for caption_word, coco_class in recall_after:
            classes_to_count[coco_class]["recall_after"] += 1

        recall_after_total += len(recall_after)
        recall_before_total += len(recall_before)
        h_before_total += len(h_before)
        h_after_total += len(h_after)

        # if len(h_before) > len(h_after):
        #   check_results.append(coco_img)

        # Calculate hallucination and recall intersections
        h_before_classes = set([ele[1] for ele in h_before])
        h_after_classes = set([ele[1] for ele in h_after])
        recall_before_classes = set([ele[1] for ele in recall_before])
        recall_after_classes = set([ele[1] for ele in recall_after])
        h_intersect_total += len(h_before_classes.intersection(h_after_classes))
        recall_intersect_total += len(
            recall_before_classes.intersection(recall_after_classes)
        )

        # if len(recall_before_classes) - len(recall_after_classes.intersection(recall_before_classes)) > 0:
        if (
            len(h_after_classes) - len(h_before_classes) < 0
            and len(recall_after_classes) >= len(recall_before_classes)
            and len(h_after_classes)
            - len(h_after_classes.intersection(h_before_classes))
            == 0
        ):
            examples_good.append(
                (
                    coco_img,
                    h_after_classes - h_before_classes.intersection(h_after_classes),
                )
            )

        new_h_total += len(h_after_classes) - len(
            h_before_classes.intersection(h_after_classes)
        )
        lost_h_total += len(h_before_classes) - len(
            h_before_classes.intersection(h_after_classes)
        )

        if len(h_after) == 0:
            h_free_images[1] += 1
        if len(h_before) == 0:
            h_free_images[0] += 1

        # Recall lost - describes the ground truth words that no longer show up as a result of this intervention method. We want this number to be as high as possible
        recall_lost_total += len(recall_before_classes) - len(
            recall_after_classes.intersection(recall_before_classes)
        )
        recall_gained_total += len(recall_after_classes) - len(
            recall_after_classes.intersection(recall_before_classes)
        )
    return dict(
        h_free_images=h_free_images,
        hallucinations=[
            h_before_total,
            h_after_total,
            h_before_total / (h_before_total + recall_before_total),
            h_after_total / (h_after_total + recall_after_total),
        ],
        recall=[recall_before_total, recall_after_total],
        hallucinations_gained=new_h_total,
        hallucinations_lost=lost_h_total,
        recall_lost=recall_lost_total,
        recall_gained=recall_gained_total,
        examples_good=examples_good,
        classes_to_count=classes_to_count,
    )
