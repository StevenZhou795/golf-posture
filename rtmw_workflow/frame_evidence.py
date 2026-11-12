"""Explain the actual selector source alongside native detector evidence."""
import numpy as np

RULES = {
    'p1': '准备姿势末端：原规则的球杆静止/起杆证据；补充杆头与球放置点距离。',
    'p2': '上杆段杆身接近水平；缺少可信杆身方向时明确标记前导臂进度回退。',
    'p3': '上杆段前导臂经过水平；人体依据，球杆框仅为辅助证据。',
    'p4': '上杆顶点：球杆方向反转与手腕高点互证；无法互证时记录原回退来源。',
    'p5': '下杆段前导臂经过水平；人体依据，球杆框仅为辅助证据。',
    'p6': '下杆段杆身接近水平；缺少可信杆身方向时明确标记前导臂进度回退。',
    'p7': '限定下杆窗口内，球离开放置点及杆头接近球互证；只有放置位置不能证明触球。',
    'p8': '击球后双臂伸展窗口内杆身接近水平；缺少可信方向时记录前导臂进度回退。',
    'p9': '送杆段手臂经过水平；人体依据，球杆框仅为辅助证据。',
    'p10': '收杆阶段双手举高且靠近头部；仍需完整挥杆覆盖与骨架质量检查。',
}


def evidence_for_frame(position, source, observation, summary):
    selected = observation.get('selected', {})
    centres = {k: None if v is None else ((np.array(v['box'][:2])+v['box'][2:])/2).tolist()
               for k, v in selected.items()}
    shaft, head = centres.get('shaft'), centres.get('head')
    ball = summary.get('ball', {})
    spot = (summary.get('impact_contact') or {}).get('ball_spot') or ball.get('p1_xy')
    angle = distance = edge_distance = None
    if observation.get('pair_usable') and shaft is not None and head is not None:
        vector = np.array(head)-shaft
        angle = float(np.degrees(np.arctan2(abs(vector[1]), abs(vector[0]))))
    if head is not None and spot is not None:
        from .swingtrace import box_distance
        distance = float(np.linalg.norm(np.array(head)-spot))
        edge_distance = box_distance(np.asarray(spot), selected['head']['box'])
    return dict(rule=RULES[position], selector_source=source,
        observation_kind='bounding_box_centre_proxies',
        raw_selected_boxes=selected, centres_px=centres,
        association_reason=observation.get('reason'),
        grip_anchor_source=observation.get('grip_anchor_source'),
        raw_pair_usable=bool(observation.get('pair_usable')),
        shaft_proxy_tilt_from_horizontal_deg=angle,
        ball_placement_px=spot,
        ball_placement_source=ball.get('source', 'yoloe' if spot is not None else 'unavailable'),
        head_centre_to_ball_px=distance, head_box_to_ball_px=edge_distance,
        impact_diagnostics=summary.get('impact_contact') if position == 'p7' else None,
        caveat='Raw detector evidence precedes track rejection. Bounding-box confidence is not landmark accuracy; proximity alone does not confirm contact.')


def enrich_manifest(manifest, observations, summary):
    by_frame = {row['frame_index']: row for row in observations}
    for row in manifest['frames']:
        row['evidence'] = evidence_for_frame(row['position'], row['source'],
            by_frame.get(row['frame_index'], {}), summary)
        # Keep the legacy evidence enum consistent with identified_frames.npz.
        # 'complete' covers the available 17+2 columns, not five physical markers.
        row['club_observation_kind'] = ('box_centre_pair' if row['evidence']['raw_pair_usable']
                                       and row.get('club_evidence') == 'complete'
                                       else 'partial_missing_or_rejected_boxes')
    manifest['club_backend'] = 'swingtrace-yolo11m'
    return manifest
