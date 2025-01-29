import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 允许加载高分辨率图片
Image.MAX_IMAGE_PIXELS = None

# 加载模型
models = {
    'XGBoost': joblib.load('1.29_xgb.pkl')
}

# 页面配置
st.set_page_config(
    page_title="热舒适度预测系统",
    page_icon="🌡️",
    layout="wide"
)

# ================= 侧边栏输入模块 =================
with st.sidebar:
    st.header("⚙️ 参数输入面板")

    # 第一层级：Basic Identifiers
    st.subheader("1. 基础标识")
    season = st.selectbox(
        "季节",
        ["冬季 (0)", "夏季 (1)", "过渡季 (2)"],
        index=0
    )

    # 第三层级：Subject's Personal Information
    st.subheader("3. 人员信息")
    col1, col2 = st.columns(2)
    with col1:
        sex = st.radio("性别", ["女 (0)", "男 (1)"], index=0)
    with col2:
        age_group = st.selectbox(
            "年龄段",
            ["<18 (0)", "18-30 (1)", "31-40 (2)", 
             "41-50 (3)", "51-60 (4)", ">61 (5)"],
            index=1
        )
    
    col3, col4 = st.columns(2)
    with col3:
        height = st.number_input("身高 (cm)", 100, 250, 170)
    with col4:
        weight = st.number_input("体重 (kg)", 30, 150, 65)

    # 第四层级：Subjective Thermal Comfort Information
    st.subheader("4. 热舒适参数")
    clothing = st.number_input("服装热阻 (clo)", 0.0, 2.0, 1.0, 0.1)
    metabolic = st.number_input("代谢率 (met)", 0.5, 4.0, 1.2, 0.1)

    # 第五层级：Indoor Physical Parameters
    st.subheader("5. 室内环境参数")
    input_mode = st.radio(
        "输入模式", 
        ["手动输入", "随机生成30组", "随机生成50组", "随机生成100组"]
    )

# ================= 数据处理模块 =================
def generate_data():
    """生成输入数据框"""
    # 解析编码值
    codes = {
        'Season': int(season.split("(")[1].replace(")", "")),
        'Sex': int(sex.split("(")[1].replace(")", "")),
        'Age Group': int(age_group.split("(")[1].replace(")", "")),
        'Height (cm)': height,
        'Weight (kg)': weight,
        'Clothing (clo)': clothing,
        'Metabolic (met)': metabolic
    }

    # 生成环境参数
    if "手动" in input_mode:
        temp = st.number_input("空气温度 (°C)", 10.0, 40.0, 25.0)
        humidity = st.number_input("相对湿度 (%)", 0.0, 100.0, 50.0)
        velocity = st.number_input("空气流速 (m/s)", 0.0, 5.0, 0.1)
        env_params = [[temp, humidity, velocity]]
    else:
        n_samples = int(input_mode.split("生成")[1].replace("组", ""))
        np.random.seed(42)
        env_params = np.column_stack([
            np.round(np.random.uniform(18, 32, n_samples), 1),
            np.round(np.random.uniform(30, 80, n_samples), 1),
            np.round(np.random.uniform(0, 1.5, n_samples), 2)
        ])

    # 构建数据框
    df = pd.DataFrame(env_params, columns=[
        'Temperature (°C)', 'Humidity (%)', 'Velocity (m/s)'
    ])
    
    # 添加固定参数
    for col, val in codes.items():
        df[col] = val

    # 调整列顺序
    feature_order = [
        'Season', 'Sex', 'Age Group', 'Height (cm)', 'Weight (kg)',
        'Clothing (clo)', 'Metabolic (met)',
        'Temperature (°C)', 'Humidity (%)', 'Velocity (m/s)'
    ]
    return df[feature_order]

# ================= 主界面显示模块 =================
st.title("🏢 建筑热舒适度智能预测系统")
df = generate_data()

# 输入数据展示
with st.expander("📥 查看输入数据", expanded=True):
    st.dataframe(df.style.format("{:.1f}"), height=300)
    st.download_button(
        label="下载输入数据",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='input_data.csv'
    )

# ================= 预测分析模块 =================
st.header("🔮 预测分析")
selected_model = st.selectbox("选择预测模型", list(models.keys()))

if st.button("开始预测"):
    try:
        model = models[selected_model]
        
        # 执行预测
        with st.spinner("预测进行中，请稍候..."):
            predictions = model.predict(df)
            proba = model.predict_proba(df) if hasattr(model, "predict_proba") else None

        # 构建结果数据框
        results_df = df.copy()
        results_df["预测结果"] = predictions
        # 定义舒适度评价的映射关系
        comfort_mapping = {
            0: "无需改变",
            1: "希望更暖",
            2: "希望更凉"
        }
        # 使用 map 函数将预测结果映射为舒适度评价
        results_df["舒适度评价"] = results_df["预测结果"].map(comfort_mapping)

        # 显示预测结果
        with st.expander("📊 查看详细预测结果", expanded=True):
            # 条件格式
            def highlight_tp(val):
                colors = {0: '#e6f3ff', 1: '#ffe6e6', 2: '#e6ffe6'}
                return f'background-color: {colors.get(val, "")}'
            
            styled_df = results_df.style.applymap(highlight_tp, subset=["预测结果"])
            st.dataframe(styled_df, height=300)

        # 可视化分析
        st.subheader("📈 分析图表")
        col1, col2 = st.columns(2)

        with col1:
            # 预测结果分布
            fig1 = plt.figure(figsize=(8, 6))
            results_df["舒适度评价"].value_counts().plot.pie(
                autopct="%1.1f%%",
                colors=["#66b3ff", "#ff9999", "#99ff99"],
                startangle=90
            )
            plt.title("预测结果分布")
            plt.ylabel("")
            st.pyplot(fig1)

        with col2:
            # 温度-舒适度关系
            fig2 = plt.figure(figsize=(8, 6))
            plt.scatter(
                results_df["Temperature (°C)"],
                results_df["预测结果"],
                c=results_df["预测结果"],
                cmap="coolwarm",
                alpha=0.7
            )
            plt.colorbar(ticks=[0, 1, 2]).set_ticklabels(["无需改变", "希望更暖", "希望更凉"])
            plt.xlabel("温度 (°C)")
            plt.ylabel("热舒适偏好")
            plt.grid(linestyle="--", alpha=0.3)
            st.pyplot(fig2)

        # 下载结果
        st.download_button(
            label="下载完整预测结果",
            data=results_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig'),
            file_name=f'predictions_{selected_model}.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"预测失败：{str(e)}")
        st.error("可能原因：\n1. 输入数据格式错误\n2. 模型文件缺失\n3. 特征列不匹配")