import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('test_data_prompt.csv')

# 提取 'image_path' 列中 '\' 前的所有内容
df['xingwei'] = df['image_path'].str.split('\\').str[0]

# 打印处理后的 DataFrame
print(df[['xingwei']])

wajue_question_ori = """
You are an intelligent assistant capable of identifying road occupancy or excavation activities in images. Please analyze the image to determine whether any ongoing construction or excavation activities are present. If such elements are clearly visible, return {"result": "yes", "bounding_boxes": [[xmin1, ymin1, xmax1, ymax1], ...]}, where the coordinates are based on a 1000x1000 pixel reference frame. Otherwise, return {"result": "no"}. Ignore normal traffic, pedestrians, highway guardrails, and non-construction-related objects.
"""

wajue_question = """
**Role:**
You are an intelligent assistant capable of accurately identifying road occupation or excavation activities in images.

**Task:**
Analyze the provided image and determine whether there are vehicles currently engaged in road occupation or excavation work.
The focus is on identifying *ongoing occupation or excavation activities*, not merely the presence of vehicles.

**To be recognized as an occupation or excavation activity, the following three conditions must all be met:**

1. The occupation or excavation activity itself is visibly taking place.
2. The surrounding area shows clear construction-related signs or obstacles, such as fences, traffic cones, or piles of soil.
   *(Note: Do not confuse ordinary road obstacles with construction-related ones.)*
3. There are people around the vehicles directing or participating in the work.

**Exclusion criteria:**

1. Ignore large vehicles that are parked or driving within safe zones and not participating in construction.
2. Ignore buildings, pedestrians, toll booths, and road dividers.
3. **Image quality limitation:**
   If the image is too blurry, obscured, or poorly lit to make an accurate judgment, respond with **“no.”**

"""

# 井盖缺失
jinggai_question_ori = """
You are an intelligent assistant capable of accurately identifying instances of missing or removed manhole covers on roads or within the road land area in images. Your task is to detect whether there is a manhole cover absence incident and return the coordinate range of the missing position in the image. Please analyze the image and determine whether there is a situation where the manhole cover has been removed, exposing the manhole opening. Focus on the locations on the road surface where the manhole cover should be but is now clearly missing (such as circular or square voids). Ignore the manhole covers that are already properly covered, pedestrians, vehicles (including parked vehicles). 
"""
jinggai_question = """
You are an intelligent assistant capable of accurately identifying instances of missing or removed manhole covers on roads or within the road land area in images. Your task is to detect whether there is a manhole cover absence incident and return the coordinate range of the missing position in the image. Please analyze the image and determine whether there is a situation where the manhole cover has been removed, exposing the manhole opening. Focus on the locations on the road surface where the manhole cover should be but is now clearly missing (such as circular or square voids). Ignore the manhole covers that are already properly covered, pedestrians, vehicles (including parked vehicles). 
"""
# 设置非公路标志
gongbiao_question_ori = """
You are an intelligent assistant capable of accurately identifying the placement of non-standard highway signs on or along the road in images. Your task is to detect if there are any non-standard highway signs set up and return their locations within the image. Analyze the image to determine whether non-highway signs, such as banners, billboards, private indicators, or other signs clearly not established by traffic management authorities, are placed on or alongside the roads. Please ignore pedestrians, vehicles (including parked cars), and legitimate road appurtenances (like lampposts, traffic signs, surveillance poles, guardrails, etc.). If you cannot clearly discern the signage due to camera angles or lighting issues, please return "no" to avoid incorrect judgments. Also, return "no" if the sign is clearly part of official traffic installations. If you detect the presence of non-standard highway signs, please return: {"result": "yes", "bounding_boxes": [[xmin1, ymin1, xmax1, ymax1], ...]}, where the coordinates are based on a 1000x1000 pixel reference frame. Otherwise, please return: {"result": "no"}.
"""
gongbiao_question = """
Role: You are an intelligent assistant with the ability to recognize road signs or billboards. You can accurately extract and analyze the text content.
Task: Please identify the signs or billboards on the road (please note: this refers to the road itself, not the buildings beside the road. If the sign is a common one on buildings, please ignore it! Also, ignore vehicle advertisements or signs). Extract the text content and determine whether it is related to "public affairs" or "personal affairs":
Words related to "personal affairs" include: "Welcome to **", "Advertisement of **", "Vehicle Maintenance of **", "Recruitment of **", etc.
Words related to "public affairs" include: 1) Words related to "transportation" (such as "Maximum Load of **", "Prohibition of **", "Drunk Driving of **", "Transportation of **", "Drive **", "Fasten Seat Belt", "Section of **", "Be Careful of **", etc.); 
2) Place names (such as Beijing, Shanghai, Xicheng District, Yao Guantun, Huangcun, etc.); 
3) Indicative words (such as "Parking Lot of **", "Gas Station of **", etc.)
Note: There may be text annotations related to the road name in the upper left corner of the picture. These are not related to the recognition task, please ignore them! Do not recognize them as text on the sign! If the text in the picture is difficult to recognize due to the shooting angle, light, or blurriness, or if you are unsure whether the text comes from an unofficial source, please reply "no" in all cases.
Output: If the text is of personal affairs nature, return: {"Result": "yes",
"Bounding Box": [[xmin1, ymin1, xmax1, ymax1]... ]},
"Content": The extracted text content }
If related to transportation or official business, reply:
{ "Result": "no",
"Content": The extracted text content } 
- The coordinates should be based on a 1000x1000 image size.
"""
# 设置悬挂物
xuangua_question_ori = """
You are an intelligent assistant capable of accurately identifying the presence of illegal hanging objects over the road or within highway premises in images. Your task is to detect if there are any non-standard, illegal hanging objects and return their locations within the image.

If you detect the presence of illegal hanging objects over the road or within the highway premises, please return:
{"result": "yes", "bounding_boxes": [[xmin1, ymin1, xmax1, ymax1], ...]}
where the coordinates are based on a 1000x1000 pixel reference frame.

Otherwise, please return:
{"result": "no"}.
"""
xuangua_question = """
Role: You are an intelligent assistant. Your task is to detect any behaviors that may endanger road safety, such as installing pipes or hanging items on road infrastructure, and return the location of such items in the image.
Task: Analyze the image and determine if there are any illegal items hanging above the road, such as ropes, decorations, or other non-road infrastructure items.
Please ignore the following: 1. Pedestrians, vehicles (including parked vehicles), and legal road facilities (such as traffic signals, traffic signs, street lamp posts, surveillance cameras, traffic guidance equipment, cables, wires, etc.) 2. Ignore items hanging on roadside buildings. 3. Ignore roadside branches, leaves, and other debris. 4. Ignore some minor hanging objects, such as a small piece of rope hanging from a bridge (if it does not endanger road safety)
Note: If there are words on the item, extract the text content and analyze its nature. If it is related to traffic, public slogans, or place names, ignore it. If the image is blurry, or due to rain, fog, or lighting reasons, the view is obstructed, and it is impossible to clearly see the image, reply "no" to avoid making a wrong judgment.
"""
# 堆放物品
wupin_question_ori = """
You are an AI assistant capable of detecting the stacking or placement of objects on the road or roadside areas. If these stacked or placed objects are clearly visible, return {"result": "yes", "bounding_boxes": [[xmin1, ymin1, xmax1, ymax1], ...]}, with the coordinates converted to a 1000x1000 pixel reference frame. Otherwise, return {"result": "no"}. Ignore normal traffic, pedestrians, vegetation, and unrelated roadside facilities.
"""
wupin_question = """
Role: You are an intelligent assistant capable of accurately identifying the act of stacking items on or within the road area.
Task: Please analyze the picture and determine whether there are any items stacked on the road or within the road area. The focus is on identifying the actual act of stacking or placing items, rather than vehicles, pedestrians, guardrails, utility poles, or road obstacles.
Please ignore the following items:
1. Items present at a construction site during construction
2. Isolation barriers, roadblocks, crash barrels, etc. used for guiding traffic, warning, or separating areas
Note: If a vehicle is identified, please return "no". If you are unsure whether this behavior constitutes stacking items, please return "no" to avoid incorrect judgments.
"""
# 摆设摊位

baitan_question_ori = """
You are an AI assistant capable of identifying the presence of illegal street vending or temporary vendor activities on the road or within road-use areas. If such activity is detected, return:
{"result": "yes", "bounding_boxes": [[xmin1, ymin1, xmax1, ymax1], ...]},
where coordinates are based on a 1000x1000 pixel reference frame.
Otherwise, return:
{"result": "no"}.
"""
baitan_question = """
Role: You are an artificial intelligence assistant capable of identifying illegal stall setups or temporary street vendors' activities on roads or within road usage areas.
Task: Please analyze the image to determine if there are any obvious signs of roadside selling, mobile stalls, or temporary booths occupying roads or sidewalks. These activities must include the set-up of stalls and the presence of vendors, and both must be present simultaneously to be recognized!
Note: If you are unsure whether this behavior constitutes a set-up stall, please return 'no' to avoid incorrect judgment. If the behavior does not occur in the road area, also return 'no'.
"""

model_result = 'Output : If the above behavior can be identified, then the following result will be returned: {"result": "yes", "bounding_boxes": [[xmin1, ymin1, xmax1, ymax1], ...]}, where the coordinates have been converted to a reference coordinate system of 1000x1000 pixels. Otherwise, return {"result": "no"}.'

# 定义映射字典
prompt_map = {
    'wupin': wupin_question_ori,
    'gongbiao': gongbiao_question_ori,
    'wajue': wajue_question_ori,
    'xuangua': xuangua_question_ori,
    'baitan': baitan_question_ori,
    'jinggai': jinggai_question_ori,
}

# 新建 prompt_ori 列
df['prompt_ori'] = df['xingwei'].map(prompt_map)

df.to_csv('test_data_prompt_deal.csv', index=False)

