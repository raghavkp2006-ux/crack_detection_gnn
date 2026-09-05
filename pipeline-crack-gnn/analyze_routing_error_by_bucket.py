"""
Analyze Routing Error Breakdown by Morphology Bucket
=====================================================

This script calculates routing accuracy and error rates broken down by
ground-truth crack morphology category (Long_Elongated, Thin_Fine_Fissure,
Branched_Complex).
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

ROUTE_TARGET = {
    'Long_Elongated': 'M4',
    'Thin_Fine_Fissure': 'M4',
    'Branched_Complex': 'M1'
}


def analyze_routing_error_by_bucket(per_image_records):
    buckets = {}
    total_correct_routes = 0
    total_n = len(per_image_records)

    for bucket_name in ROUTE_TARGET:
        subset = [r for r in per_image_records if r['gt_category'] == bucket_name]
        n_images = len(subset)
        if n_images == 0:
            continue

        n_correctly_routed = sum(
            1 for r in subset if ROUTE_TARGET[r['pred_category']] == ROUTE_TARGET[bucket_name]
        )
        n_category_matched = sum(
            1 for r in subset if r['pred_category'] == bucket_name
        )

        total_correct_routes += n_correctly_routed

        buckets[bucket_name] = {
            'n_images': n_images,
            'n_correctly_routed': n_correctly_routed,
            'routing_accuracy': round(n_correctly_routed / n_images, 4),
            'routing_error_rate': round(1 - n_correctly_routed / n_images, 4),
            'n_category_matched': n_category_matched,
            'category_accuracy': round(n_category_matched / n_images, 4),
            'category_error_rate': round(1 - n_category_matched / n_images, 4),
        }

    return {
        'overall': {
            'total_n': total_n,
            'total_correct_routes': total_correct_routes,
            'error_rate': round(1 - total_correct_routes / total_n, 4),
        },
        'by_bucket': buckets,
    }


if __name__ == '__main__':
    per_image_path = os.path.join(RESULTS_DIR, 'router_per_image_categories.json')
    if not os.path.exists(per_image_path):
        raise FileNotFoundError(
            f'{per_image_path} not found. Add per-image gt_category/pred_category '
            'logging to run_noncircular_router_validation.py and re-run it first -- '
            'see this module\'s docstring for the exact code to add.'
        )
    with open(per_image_path) as f:
        per_image_records = json.load(f)

    result = analyze_routing_error_by_bucket(per_image_records)
    out_path = os.path.join(RESULTS_DIR, 'routing_error_by_bucket.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved routing error breakdown to {out_path}')
    print(json.dumps(result, indent=2))
