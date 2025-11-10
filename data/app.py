import streamlit as st
import pandas as pd
from pathlib import Path
from deep_translator import GoogleTranslator

# -----------------------------
# 配置文件路径和图片根目录
# -----------------------------
CSV_FILE = "./result_youxiao.csv"       # 改成你的实际 CSV 文件路径
IMAGE_BASE = "./pic_pack/"     # 图片所在根目录

# -----------------------------
# 读取 CSV 数据
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(CSV_FILE)

df = load_data()

# -----------------------------
# 页面设置
# -----------------------------
st.set_page_config(page_title="图片识别结果浏览器", layout="wide")
st.title("📑 图片识别结果浏览器")

# -----------------------------
# 侧边栏配置
# -----------------------------
st.sidebar.header("⚙️ 设置")
translate_flag = st.sidebar.checkbox("是否翻译为中文", value=False)

# -----------------------------
# 搜索栏
# -----------------------------
search = st.text_input("🔍 输入图片文件名搜索", "")
if search:
    data = df[df["image_path"].str.contains(search, case=False, na=False)]
else:
    data = df

# -----------------------------
# 翻页状态
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = 0

page_size = 1
total_pages = (len(data) - 1) // page_size + 1

# -----------------------------
# 页码选择器（现在在上方）
# -----------------------------
page_selector = st.number_input(
    "跳转到页码",
    min_value=1,
    max_value=total_pages,
    value=st.session_state.page + 1,
    step=1
)
st.session_state.page = page_selector - 1

st.markdown("---")  # 分隔线

# -----------------------------
# 翻页按钮：左右固定在界面边缘（现在在下方）
# -----------------------------
cols = st.columns([1, 8, 1])  # 左1，中8，右1

with cols[0]:
    if st.button("⬅️ 上一条", key="prev"):
        if st.session_state.page > 0:
            st.session_state.page -= 1

with cols[2]:
    if st.button("下一条 ➡️", key="next"):
        if st.session_state.page < total_pages - 1:
            st.session_state.page += 1

# -----------------------------
# 当前条目数据
# -----------------------------
row = data.iloc[st.session_state.page]

# -----------------------------
# 翻译函数
# -----------------------------
def maybe_translate(text, do_translate):
    if not do_translate:
        return text
    try:
        return GoogleTranslator(source="en", target="zh-CN").translate(text)
    except Exception:
        return text + "\n\n⚠️ 翻译失败"

# -----------------------------
# 中间显示图片和文字
# -----------------------------
middle_left, middle_right = st.columns([1, 2])

with middle_left:
    img_path = Path(IMAGE_BASE) / row["image_path"]
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.warning(f"⚠️ 图片未找到: {img_path}")

with middle_right:
    st.subheader("📌 模型分析结果")
    st.markdown(maybe_translate(row["model_result"], translate_flag))

    st.subheader("📝 提示词")
    st.markdown(f"```text\n{maybe_translate(row['prompt_text'], translate_flag)}\n```")

# -----------------------------
# 页码显示
# -----------------------------
st.write(f"第 {st.session_state.page+1} / {total_pages} 条")
