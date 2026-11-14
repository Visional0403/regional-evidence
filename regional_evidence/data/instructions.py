from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from .constants import CATEGORY_NAMES_ZH, COLUMN_NAMES_ZH, REGION_NAMES_ZH

GeneratorFunction = Callable[[dict, random.Random], tuple[str, str]]
VEHICLE_KEYS = ('car', 'van', 'truck', 'bus', 'bicycle', 'motor', 'tricycle', 'awning_tricycle')
OBJECT_KEYS = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning_tricycle', 'bus', 'motor', 'others')
COLUMNS = ('left', 'center', 'right')
VEHICLE_SCOPE_OPTIONS = ('本项目车辆总数固定为car、van、truck、bus、bicycle、motor、tricycle和awning_tricycle八类有效标注之和', '车辆统计口径只包含car、van、truck、bus、bicycle、motor、tricycle、awning_tricycle八类有效框', '计算车辆时合计car、van、truck、bus、bicycle、motor、tricycle及awning_tricycle，不纳入pedestrian或people')
GRID_RULE_OPTIONS = ('空间位置使用图像平面九宫格：画面宽、高各三等分，bbox中心点决定归属；方位只能解释为图像坐标，不能改写成交叉口、入口、车道等道路语义位置', '九宫格由图像宽度和高度分别三等分得到，每个bbox按中心点落入唯一格子；回答的是画面方位，不是路口或车道名称', '采用图像坐标九宫格，横纵方向各分成三段并以目标框中心分配区域；不要用道路入口、交叉口或行驶方向替代')
COLUMN_RULE_OPTIONS = ('left、center、right是画面宽度方向的三个等宽区域，不表示道路行驶方向', '左中右统计按图像宽度三等分，对应left、center、right，与车流方向无关', '三列是画面上的等宽纵向区域left、center、right，不是道路的左转、直行或右转方向')
COUNT_FACT_RULE_OPTIONS = ('只回答有效标注的类别与数量，不描述密度、拥堵、交通压力、速度或其他场景推断', '答案限于标注计数，不加入道路状态、车辆速度或主观交通判断', '仅陈述能够从有效标注复算的数量事实，不扩展拥堵、密度或场景原因')
SPATIAL_FACT_RULE_OPTIONS = ('只回答车辆数量和由bbox中心点计算的图像方位，不推断拥堵、交通压力、道路结构或未来变化', '答案限于车辆计数及图像坐标分布，不解释道路状况、通行压力或后续趋势', '仅陈述可由车辆框中心点复算的空间事实，不添加拥堵、车速或道路语义判断')
OBJECT_JSON_SCHEMA = '{"object_count":{"pedestrian":0,"people":0,"bicycle":0,"car":0,"van":0,"truck":0,"tricycle":0,"awning_tricycle":0,"bus":0,"motor":0,"others":0,"total_valid_objects":0}}'
SPATIAL_JSON_SCHEMA = '{"vehicle_distribution":{"left":0,"center":0,"right":0},"vehicle_grid":{"top-left":0,"top-center":0,"top-right":0,"center-left":0,"center":0,"center-right":0,"bottom-left":0,"bottom-center":0,"bottom-right":0},"largest_vehicle_cluster_regions":[],"busiest_vehicle_columns":[]}'

@dataclass(frozen=True)
class GeneratorSpec:
    family: str
    name: str
    generate: GeneratorFunction
REGISTRY: dict[str, dict[str, GeneratorSpec]] = defaultdict(dict)
TASK_ORDER = (('structured', 'object_statistics'), ('natural', 'vehicle_composition'), ('natural', 'pedestrian_statistics'), ('structured', 'spatial_distribution'), ('natural', 'spatial_comparison'))

def register(family: str, name: str) -> Callable:

    def decorator(function: GeneratorFunction) -> GeneratorFunction:
        if name in REGISTRY[family]:
            raise ValueError(f'Duplicate generator: {family}/{name}')
        REGISTRY[family][name] = GeneratorSpec(family, name, function)
        return function
    return decorator

def compose_question(rng: random.Random, openings: tuple[str, ...], intents: tuple[str, ...], required: tuple[str, ...]) -> str:
    clauses: list[str] = []
    if rng.random() < 0.72:
        clauses.append(rng.choice(openings))
    clauses.append(rng.choice(intents))
    if rng.random() < 0.62:
        clauses.append(rng.choice(('答案将与scene graph逐项核对', '所有要求项目都需要明确给出', '请严格遵守后述统计口径', '结果用于自动化事实评测', '不要省略题目指定的字段或类别', '请让输出能够直接复核')))
    clauses.extend(required)
    return '，'.join(clause.strip('，。？') for clause in clauses) + '。'

def _object_payload(scene: dict) -> dict:
    payload = {key: scene['category_counts'][key] for key in OBJECT_KEYS}
    payload['total_valid_objects'] = scene['totals']['valid_objects']
    return payload

def _vehicle_payload(scene: dict) -> dict:
    payload = {'total': scene['totals']['vehicles']}
    payload.update({key: scene['category_counts'][key] for key in VEHICLE_KEYS})
    return payload

def _zh_list(values: list[str], names: dict[str, str]) -> str:
    return '、'.join(names[value] for value in values)

def _vehicle_detail_text(scene: dict) -> str:
    return '、'.join(f"{CATEGORY_NAMES_ZH[key]}{scene['category_counts'][key]}辆" for key in VEHICLE_KEYS)

@register('structured', 'object_statistics')
def object_statistics(scene: dict, rng: random.Random) -> tuple[str, str]:
    question = compose_question(rng, ('依据VisDrone有效标注', '针对当前无人机画面', '为便于程序核验'), ('统计全部目标类别', '输出各类目标的精确数量', '核对本帧目标组成'), ('score=0或category=0的忽略区域不计入答案', f'严格按照以下完整模板返回一个JSON对象：{OBJECT_JSON_SCHEMA}', '模板中的0只是类型示例，必须替换为本图对应的非负整数；即使某一类别数量为0也必须保留该字段；不得增加、删除、改名、交换或省略任何字段', '输出必须以左花括号开始、右花括号结束，不使用Markdown代码块，不在JSON前后添加解释、标题或注释；所有object_count内部值都必须是整数', rng.choice(COUNT_FACT_RULE_OPTIONS)))
    return (question, json.dumps({'object_count': _object_payload(scene)}, ensure_ascii=False, separators=(',', ':')))

@register('natural', 'vehicle_composition')
def vehicle_composition(scene: dict, rng: random.Random) -> tuple[str, str]:
    question = compose_question(rng, ('根据检测标注', '查看这张航拍图后', '面向车辆数量复核'), ('说明车辆总数与车型组成', '逐类报告画面中的车辆数量', '给出完整车辆计数结果'), (rng.choice(VEHICLE_SCOPE_OPTIONS), '回答必须同时给出车辆总数和八类车辆各自数量，数量为0的类别也不能省略', '使用一到两句自然语言作答，所有数字必须与标注一致', rng.choice(COUNT_FACT_RULE_OPTIONS)))
    total = scene['totals']['vehicles']
    details = _vehicle_detail_text(scene)
    answer = rng.choice((f'按有效标注统计，画面中共有{total}辆车，具体为{details}。', f'车辆总数为{total}辆；八类车辆分别是{details}。'))
    return (question, answer)

@register('natural', 'pedestrian_statistics')
def pedestrian_statistics(scene: dict, rng: random.Random) -> tuple[str, str]:
    question = compose_question(rng, ('按照本帧有效标注', '从人员目标统计角度', '为核验行人相关数量'), ('分别统计pedestrian和people', '报告两类人员目标及其合计', '给出行人相关目标组成'), ('行人相关目标固定定义为pedestrian与people两类有效标注之和', '回答必须包含pedestrian数量、people数量和两类合计，不推测遮挡目标', '使用一句自然语言作答，数字必须精确', rng.choice(COUNT_FACT_RULE_OPTIONS)))
    pedestrian = scene['category_counts']['pedestrian']
    people = scene['category_counts']['people']
    total = scene['totals']['pedestrian_related']
    answer = rng.choice((f'画面中有{pedestrian}个pedestrian标注和{people}个people标注，行人相关目标合计{total}个。', f'按标注统计，pedestrian为{pedestrian}个、people为{people}个，两类共{total}个。'))
    return (question, answer)

@register('structured', 'spatial_distribution')
def spatial_distribution(scene: dict, rng: random.Random) -> tuple[str, str]:
    question = compose_question(rng, ('依据车辆bbox中心点', '对当前画面进行坐标统计', '为自动评测车辆空间分布'), ('输出车辆的三分区和九宫格数量', '生成车辆空间分布结果', '核对车辆所在图像区域'), (rng.choice(VEHICLE_SCOPE_OPTIONS), rng.choice(GRID_RULE_OPTIONS), rng.choice(COLUMN_RULE_OPTIONS), '并列最大区域必须全部保留；没有车辆时两个最大区域列表均为空', f'严格按照以下完整模板返回一个JSON对象：{SPATIAL_JSON_SCHEMA}', '模板中的0只是类型示例，必须替换为本图对应的非负整数；vehicle_distribution只能包含left、center、right三个整数键，vehicle_grid只能包含top-left、top-center、top-right、center-left、center、center-right、bottom-left、bottom-center、bottom-right九个整数键', 'largest_vehicle_cluster_regions必须是仅含上述九宫格英文名称的字符串数组；busiest_vehicle_columns必须是仅含left、center、right的字符串数组；不得增加、删除、改名或交换任何字段', '输出必须以左花括号开始、右花括号结束，不使用Markdown代码块，不在JSON前后添加解释、标题或注释', rng.choice(SPATIAL_FACT_RULE_OPTIONS)))
    spatial = scene['spatial']
    answer = {'vehicle_distribution': spatial['vehicle_distribution'], 'vehicle_grid': spatial['vehicle_grid'], 'largest_vehicle_cluster_regions': spatial['largest_vehicle_cluster_regions'], 'busiest_vehicle_columns': spatial['busiest_vehicle_columns']}
    return (question, json.dumps(answer, ensure_ascii=False, separators=(',', ':')))

@register('natural', 'spatial_comparison')
def spatial_comparison(scene: dict, rng: random.Random) -> tuple[str, str]:
    question = compose_question(rng, ('根据车辆框中心点统计', '从图像坐标分布看', '为复核车辆空间位置'), ('比较左中右车辆数量并指出九宫格聚集位置', '说明车辆在画面中的精确分布', '给出车辆区域比较结论'), (rng.choice(VEHICLE_SCOPE_OPTIONS), rng.choice(GRID_RULE_OPTIONS), rng.choice(COLUMN_RULE_OPTIONS), '回答必须给出左、中、右三个数量，以及所有并列最多列和所有并列的最大车辆聚集九宫格区域；没有车辆时明确说明', '使用两到三句自然语言作答，不使用交通压力或拥堵等解释', rng.choice(SPATIAL_FACT_RULE_OPTIONS)))
    spatial = scene['spatial']
    distribution = spatial['vehicle_distribution']
    if scene['totals']['vehicles'] == 0:
        return (question, '画面左侧、中部和右侧的车辆数均为0，九宫格内也没有车辆，因此不存在车辆最多列或最大车辆聚集区域。')
    busiest_columns = _zh_list(spatial['busiest_vehicle_columns'], COLUMN_NAMES_ZH)
    cluster_regions = _zh_list(spatial['largest_vehicle_cluster_regions'], REGION_NAMES_ZH)
    maximum_column_count = max(distribution.values())
    maximum_grid_count = max(spatial['vehicle_grid'].values())
    answer = rng.choice((f"画面左侧、中部和右侧分别有{distribution['left']}辆、{distribution['center']}辆和{distribution['right']}辆车，数量最多的列为{busiest_columns}，均为{maximum_column_count}辆。九宫格中车辆最多的是{cluster_regions}，均有{maximum_grid_count}辆车。", f"按画面宽度三等分，左、中、右车辆数依次为{distribution['left']}、{distribution['center']}、{distribution['right']}辆；最多列是{busiest_columns}，对应{maximum_column_count}辆。按九宫格统计，最大车辆聚集区域为{cluster_regions}，对应数量为{maximum_grid_count}辆。"))
    return (question, answer)
