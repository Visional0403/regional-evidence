from __future__ import annotations

CATEGORY_NAMES = {0: 'ignored', 1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van', 6: 'truck', 7: 'tricycle', 8: 'awning_tricycle', 9: 'bus', 10: 'motor', 11: 'others'}
CATEGORY_NAMES_ZH = {'pedestrian': '行人', 'people': '人员', 'bicycle': '自行车', 'car': '小汽车', 'van': '厢式车', 'truck': '卡车', 'tricycle': '三轮车', 'awning_tricycle': '带棚三轮车', 'bus': '公交车', 'motor': '摩托车', 'others': '其他目标'}
VEHICLE_CATEGORIES = frozenset({3, 4, 5, 6, 7, 8, 9, 10})
REGION_NAMES = ('top-left', 'top-center', 'top-right', 'center-left', 'center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right')
REGION_NAMES_ZH = {'top-left': '左上区域', 'top-center': '上方中央区域', 'top-right': '右上区域', 'center-left': '左侧中部区域', 'center': '画面中央区域', 'center-right': '右侧中部区域', 'bottom-left': '左下区域', 'bottom-center': '下方中央区域', 'bottom-right': '右下区域'}
COLUMN_NAMES_ZH = {'left': '左侧', 'center': '中部', 'right': '右侧'}
